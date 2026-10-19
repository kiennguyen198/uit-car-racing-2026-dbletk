"""Bounded local recovery from a stationary perception stop.

No checkpoint, world position, simulator state or stored map route. Reverse
uses only a short recently observed straight approach; there is no rear
camera, so that estimate is limited and is never treated as collision proof.
"""
from collections import deque
from dataclasses import replace
import math

import cv2
import numpy as np
from quality_guard import current_near_water_path, current_near_curve_path, current_white_mask


def heading_difference(first, second):
    return (float(first)-float(second)+180) % 360-180


class StuckRecovery:
    def __init__(self, *, allow_reverse=False):
        self.allow_reverse = bool(allow_reverse)
        self.elapsed = 0.0
        self.travel_m = 0.0
        self.history = deque()
        self.stopped_s = 0.0
        self.clear_s = 0.0
        self.phase = 'idle'
        self.phase_s = 0.0
        self.phase_distance = 0.0
        self.total_distance = 0.0
        self.attempts = 0
        self.goal_angle = 0.0
        self.reverse_heading = None
        self.reverse_limit = 0.0
        self.path_target = None
        self.path_frames = 0
        self.info = {}
        self.progress_samples = deque()
        self.progress_span_s = 0.0
        self.progress_distance_m = 0.0
        self.low_progress = False
        self.path_seen_at = -math.inf
        self.revalidation_s = 0.0
        self.revalidation_target = None
        self.revalidation_seen_at = -math.inf
        self.revalidation_cycles = 0
        self.reverse_used = False

    def _path_goal(self, path, width, yaw):
        gain = 35 if path.get('curve_recovery') else 30
        limit = 25 if path.get('curve_recovery') else 18
        goal = float(np.clip(gain*(path['target']-width/2)/(width/2)-.08*yaw,
                             -limit, limit))
        minimum = path.get('minimum_angle', 0)
        if minimum > 0: goal = max(goal, minimum)
        if minimum < 0: goal = min(goal, minimum)
        return goal

    def _clear_revalidation(self):
        self.revalidation_s = 0.0
        self.revalidation_target = None
        self.revalidation_seen_at = -math.inf

    def _renewable_current_path(self, path, mask, raw, speed, wheel, yaw):
        # A retry requires new CURRENT observations, not a cached target or
        # expiry of a timer. Renew only with paired visible edges and ground.
        if (path is None or self.reverse_used or speed >= .4 or abs(yaw) >= 10
                or path['near_clearance'] <= .30):
            return False
        h, w = mask.shape
        white = current_white_mask(cv2.resize(raw, (w, h)))
        foot = (slice(int(.97*h), int(.995*h)), slice(int(.49*w), int(.51*w)))
        if (mask[foot] == 1).mean() < .95 or white[foot].mean() > .25:
            return False
        paired = 0
        fractions = (.995, .97, .93, .89) if path.get('curve_recovery') else (.985, .93, .87, .83)
        for fraction in fractions:
            y = min(h-1, int(fraction*h))
            xs = np.flatnonzero((mask[max(0,y-2):min(h,y+3)] == 1).mean(0) > .8)
            if len(xs) < 8 or xs[0] < 3 or xs[-1] >= w-3:
                continue
            visible = []
            for x in (int(xs[0])+5, int(xs[-1])-5):
                strip = white[max(0,y-2):min(h,y+3), max(0,x-2):min(w,x+3)]
                visible.append(strip.size and strip.mean() < .40)
            paired += all(visible)
        return paired >= 2

    def _observe_progress(self, speed, dt):
        # A brief 0->1 km/h twitch is not proof that the car escaped spray.
        # This is measured travel over a local time window, not a map pose.
        self.progress_samples.append((self.elapsed, speed / 3.6 * dt, dt))
        while self.progress_samples and self.elapsed-self.progress_samples[0][0] >= 2.0:
            self.progress_samples.popleft()
        self.progress_span_s = sum(s[2] for s in self.progress_samples)
        self.progress_distance_m = sum(s[1] for s in self.progress_samples)
        self.low_progress = (self.progress_span_s >= 1.95
                             and self.progress_distance_m < .30 and speed < 2)

    def _progress_info(self):
        return dict(progress_span_s=self.progress_span_s,
                    progress_distance_m=self.progress_distance_m,
                    low_progress=self.low_progress)

    def _output(self, decision, speed, angle, reason):
        self.info = dict(phase=self.phase, reason=reason, attempts=self.attempts,
                         phase_distance_m=self.phase_distance,
                         total_distance_m=self.total_distance,
                         stopped_s=self.stopped_s,
                         reverse_enabled=self.allow_reverse,
                         revalidation_cycles=self.revalidation_cycles,
                         revalidation_s=self.revalidation_s)
        self.info.update(self._progress_info())
        return replace(decision, speed=float(speed), angle=float(angle),
                       mode='recovery_'+self.phase)

    def _recent_reverse(self, heading):
        # Retain at most 6 seconds of telemetry; choose the nearest safe
        # anchor on a nearly straight approach. Distance is a conservative
        # integral of speed, not a measured world pose or rear occupancy.
        for sample in reversed(self.history):
            distance = self.travel_m-sample['travel_m']
            if not .20 <= distance <= 1.5:
                continue
            later = [s for s in self.history if s['time'] >= sample['time']]
            if (self.elapsed-sample['time'] <= 4.5 and sample['safe']
                    and abs(heading_difference(heading, sample['heading'])) < 12
                    and all(abs(heading_difference(s['heading'], heading)) < 12
                            and not s['traffic_blocked'] for s in later)):
                return sample['heading'], min(.65, distance)
        return None

    def update(self, mask, raw, state, decision, traffic, *, dt_s, fresh=True):
        dt = float(np.clip(dt_s, .01, .25))
        speed = abs(float(state.get('Speed', 0)))
        heading = float(state.get('Heading', float('nan')))
        wheel = float(state.get('Angle', float('nan')))
        self.elapsed += dt
        if not fresh or not all(math.isfinite(x) for x in (speed, heading, wheel)):
            # Do not count unobserved motion or keep old path confirmation.
            self.progress_samples.clear()
            self.progress_span_s = self.progress_distance_m = 0.0
            self.low_progress = False
            self.path_frames = 0
            self.path_target = None
            self._clear_revalidation()
            self.info = dict(phase=self.phase, reason='stale_or_invalid_status',
                             attempts=self.attempts, revalidation_cycles=self.revalidation_cycles)
            if self.phase != 'idle':
                self.phase = 'settle'
                self.phase_s = 0.0
                return self._output(decision, 0, 0, 'stale_or_invalid_status')
            return decision
        distance = speed/3.6*dt
        self.travel_m += distance
        blocked = bool(traffic.get('pending_sign') or traffic.get('sign_conflicts')
                       or traffic.get('turn_direction') or traffic.get('crossing')
                       or traffic.get('junction_mouth_open') or traffic.get('t_junction')
                       or traffic.get('obstacles_on_road', 0)
                       or traffic.get('uncertain_side_contacts', 0)
                       or traffic.get('gap_target') is not None or traffic.get('avoid_side'))
        # Match RoadController ownership: a strong current clipped curve
        # must be the same steering/safety corridor in every layer.
        path = current_near_curve_path(mask, raw)
        if path is None:
            path = current_near_water_path(mask, raw)
        safe = bool(path and decision.confidence == 1 and decision.near_clearance > .30
                    and decision.preview_quality >= .8 and not decision.junction_suspect
                    and decision.occlusion_fraction < .025 and not blocked
                    and abs(decision.yaw_rate) < 12 and abs(wheel) < 8
                    and decision.speed > 0)
        self.history.append(dict(time=self.elapsed, travel_m=self.travel_m,
                                 heading=heading, safe=safe, traffic_blocked=blocked))
        while self.history and self.elapsed-self.history[0]['time'] > 6:
            self.history.popleft()
        self.stopped_s = self.stopped_s+dt if speed < .5 else 0.0
        self._observe_progress(speed, dt)
        self.clear_s = self.clear_s+dt if safe and speed > 2 else 0.0
        if self.clear_s > 2 and self.phase == 'idle':
            self.attempts = 0
            self.total_distance = 0.0
            self.reverse_used = False
        self.info = dict(phase=self.phase, attempts=self.attempts,
                         stopped_s=self.stopped_s, total_distance_m=self.total_distance)
        self.info.update(self._progress_info())
        self.info['revalidation_cycles'] = self.revalidation_cycles
        self.info['reason']='no_recovery_needed'
        if traffic.get('branch_owner') or traffic.get('local_recovery_owner'):
            # The committed branch already owns steering, edge checks and
            # bounded forward motion. Do not launch a second competing path.
            self.info['reason']=('shared_current_near_corridor_owned_by_traffic'
                if traffic.get('local_recovery_owner') else
                'shared_committed_branch_owned_by_traffic')
            self.phase = 'idle'
            self.phase_s = self.phase_distance = 0.0
            self.path_frames = 0
            self.path_target = None
            self._clear_revalidation()
            self.info['phase'] = 'idle'
            return decision
        if blocked:
            self._clear_revalidation()
            self.info['reason']='traffic_hazard_or_unresolved_branch_veto'
            if self.phase != 'idle':
                self.phase = 'settle'
                self.phase_s = 0
                return self._output(decision, 0, 0, 'traffic_or_stale_status_veto')
            return decision
        if (decision.mode in ('status_stale','final_status_stale','manual_assist',
                'corner_momentum_brake','fountain_physical_brake',
                'sign_speed_brake','turn_alignment_physical_brake','visibility_caution')
                or decision.edge_trend_guard):
            self._clear_revalidation()
            self.path_frames = 0
            self.path_target = None
            self.phase = 'idle'
            self.phase_s = self.phase_distance = 0.0
            self.info.update(phase='idle', reason='current_physical_or_edge_brake_veto')
            return decision
        owned_path = decision.current_path
        owner_ready = bool(path is not None and owned_path is not None and decision.speed >= 4
            and decision.mode in ('normal','corner','fountain_visible_pass',
                                  'occlusion_near_recovery','near_curve_tracking')
            and not decision.geometry_guard and not decision.steering_settle_guard
            and abs(float(owned_path.get('target',decision.target_x))-path['target']) < .06*mask.shape[1]
            and abs(wheel-decision.angle) < 4 and abs(decision.yaw_rate) < 25)
        if owner_ready and (self.phase != 'idle' or self.stopped_s >= 1.2 or self.low_progress):
            # Passing an already checked owner's command is not an attempt.
            self.phase = 'idle'
            self.phase_s = self.phase_distance = 0.0
            self.path_frames = 0
            self.path_target = None
            self._clear_revalidation()
            self.info.update(phase='idle', reason='current_corridor_return_to_owner')
            return decision
        if self.phase == 'idle':
            # Initial START/Manual waiting, legal sign stops, full lap finish
            # and a normal positive command are not assumed to be a failure.
            # A verified near-curve can intentionally command zero while
            # the delayed physical wheel reaches its requested angle. Treat
            # these visible recovery/edge-brake modes as stalled candidates;
            # current path, three matching observations, live wheel alignment,
            # obstacle/branch vetoes and the existing 0.8m/2s limits still gate
            # every forward command below. No reverse or hidden-road motion.
            recoverable = decision.mode in (
                'traffic_perception_stop','occlusion_stop','occlusion_hold',
                'mask_lost','occluded_traffic_stop','occlusion_near_recovery',
                'near_curve_tracking','spray_prealert_brake')
            if not recoverable or (self.stopped_s < 1.2 and not self.low_progress):
                return decision
            if self.attempts >= 2 or self.total_distance >= 1.2:
                if not self._renewable_current_path(path,mask,raw,speed,wheel,decision.yaw_rate):
                    self._clear_revalidation()
                    self.info['reason'] = 'recovery_budget_wait_current_edges_and_wheel'
                    return decision
                goal = self._path_goal(path,mask.shape[1],decision.yaw_rate)
                self.goal_angle = float(np.clip(goal,wheel-25*dt,wheel+25*dt))
                if abs(wheel-goal) >= 4 or abs(self.goal_angle-goal) >= 4:
                    self._clear_revalidation()
                    return self._output(decision,0,self.goal_angle,
                                        'recovery_budget_align_on_current_path')
                target = float(path['target'])
                agrees = (self.revalidation_target is not None
                    and self.elapsed-self.revalidation_seen_at <= .25
                    and abs(target-self.revalidation_target) < .06*mask.shape[1])
                self.revalidation_s = self.revalidation_s+dt if agrees else 0.0
                self.revalidation_target = target
                self.revalidation_seen_at = self.elapsed
                self.info.update(reason='recovery_budget_confirm_new_current_path',
                                 revalidation_s=self.revalidation_s)
                if self.revalidation_s < .35:
                    return decision
                # End one bounded episode only after independent fresh
                # geometry/physical checks. Reverse budgets never renew here.
                self.attempts = 0
                self.total_distance = 0.0
                self.path_frames = 0
                self.path_target = None
                self.revalidation_cycles += 1
                self._clear_revalidation()
                self.info.update(reason='recovery_budget_revalidated_current_path',
                                 attempts=0, revalidation_cycles=self.revalidation_cycles)
            if path:
                self.path_frames = self.path_frames+1 if (self.path_target is not None
                    and self.elapsed-self.path_seen_at <= .25
                    and abs(path['target']-self.path_target) < .06*mask.shape[1]) else 1
                self.path_target = path['target']
                self.path_seen_at = self.elapsed
                if self.path_frames < 3:
                    self.info['reason'] = 'low_progress_confirm_current_path'
                    return decision
                self.phase = 'align'
            else:
                self.path_frames = 0
                self.path_target = None
                reverse = self._recent_reverse(heading) if self.allow_reverse else None
                if reverse is None:
                    self.info['reason'] = 'no_visible_forward_path_or_recent_safe_reverse'
                    return decision
                self.reverse_heading, self.reverse_limit = reverse
                self.reverse_used = True
                self.phase = 'reverse'
            self.attempts += 1
            # Start steering from measured wheel angle, not zero. The car
            # has already aligned partway before a mask dropout.
            self.goal_angle = float(max(-30.0,min(30.0,wheel)))
            self.phase_s = self.phase_distance = 0.0
            self.stopped_s = 0.0
        self.phase_s += dt
        self.phase_distance += distance
        self.total_distance += distance
        if self.phase == 'reverse':
            delta = heading_difference(heading, self.reverse_heading)
            if (self.phase_s > 1.2 or self.phase_distance >= self.reverse_limit
                    or self.total_distance >= 1.2 or abs(delta) > 15 or speed > 5):
                self.phase = 'settle'; self.phase_s = 0
                return self._output(decision, 0, 0, 'reverse_limit_reached')
            # Straight recent approach only; a small heading correction,
            # with inverse yaw response in reverse, is bounded to6degrees.
            angle = float(np.clip(.4*delta, -6, 6))
            return self._output(decision, -6, angle, 'short_recent_approach_reverse')
        if self.phase in ('align', 'forward'):
            # Once Road/Fountain observes the same corridor and can already
            # advance, recovery must hand it back rather than adding a low
            # request or another alignment stop. This grants no new speed:
            # it returns the command already checked by the owning layers.
            if self.phase_s > 2 or self.phase_distance >= .8 or self.total_distance >= 1.2:
                self.phase = 'settle'; self.phase_s = 0
                return self._output(decision, 0, self.goal_angle, 'current_path_lost_or_motion_limit')
            if path is None:
                # Brake immediately on a missing current path, but keep the
                # last steering intent for re-observation instead of snapping
                # the physical wheels toward zero during one bad frame.
                self.path_frames = 0; self.path_target = None
                self.phase = 'align'
                return self._output(decision, 0, self.goal_angle, 'current_path_lost_brake_reobserve')
            width = mask.shape[1]
            w = width
            self.path_frames = (self.path_frames+1 if self.path_target is not None
                and self.elapsed-self.path_seen_at <= .25
                and abs(path['target']-self.path_target)<.06*width else 1)
            self.path_target = path['target']
            self.path_seen_at = self.elapsed
            goal = self._path_goal(path,width,decision.yaw_rate)
            self.goal_angle = float(np.clip(goal, self.goal_angle-25*dt, self.goal_angle+25*dt))
            aligned = (abs(wheel-goal) < 4 and abs(self.goal_angle-goal) < 4
                       and abs(decision.yaw_rate) < 25)
            confirmed = self.path_frames >= 3
            self.phase = 'forward' if aligned and confirmed else 'align'
            return self._output(decision, 4 if aligned and confirmed and speed < 7 else 0,
                                self.goal_angle, 'fresh_connected_near_road' if confirmed else 'confirm_same_current_path')
        if self.phase == 'settle':
            if self.phase_s >= .35 and speed < .4:
                self.phase = 'idle'
                self.path_frames = 0
                self.path_target = None
                self.history.clear()  # In-memory evidence only; logged files are preserved.
            return self._output(decision, 0, 0, 'brake_and_reobserve')
        return decision
