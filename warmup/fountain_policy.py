from speed_profile import band
"""Fountain warning + current road evidence; no stored route or hidden-road motion.

History confirms/smooths measurements. Every positive replacement command
requires a CURRENT connected corridor and visible road beside both edges.
Traffic, physical overspeed and stale-data vetoes remain downstream.
"""
from collections import deque
from fountain_memory import FountainMemory
from dataclasses import replace
import cv2
import numpy as np
from quality_guard import visible_spray_corridor, current_near_water_path, current_white_mask


class FountainPolicy:
    def __init__(self):
        self.elapsed = 0.0
        self.hits = deque()
        self.last_cue = -1e9
        self.last_close_cue = -1e9
        self.observations = deque()
        self.last_angle = None
        self.last_angle_time = -1e9
        self.braking = False
        self.info = {}
        self.memory = FountainMemory()

    @staticmethod
    def current_path(mask, raw):
        h, w = mask.shape
        image = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
        white = current_white_mask(image)
        # Explicitly reject an opaque image even if segmentation hallucinates Road.
        if white[int(.55*h):].mean() > .80:
            return None, 'opaque_camera_ground'
        # Same observed-edge fit also runs BEFORE spray; only the white-pixel
        # minimum is removed. Opaque/ground/margin/junction checks are retained.
        path = visible_spray_corridor(mask, raw, require_spray=False)
        if path is not None:
            path = dict(path, source='current_fitted_fountain_edges',
                        near_clearance=path['clearance'], target=path['centers'][1],
                        center=path['centers'][0], width=path['widths'][0],
                        target_width=path['widths'][1], flowing=True)
        else:
            path = current_near_water_path(mask, raw)
            if path is None:
                return None, 'no_current_connected_corridor'
            path = dict(path, source='current_connected_near_edges', flowing=False)
        # Count CURRENT unoccluded edge interiors, not a remembered mask or
        # isolated patch in the middle. Wide/clipped crossroads are rejected
        # by both constructors above. Interior strips exclude white edge paint.
        paired = 0
        # Verify the SAME current inlier observations used to fit the path.
        # Fixed lower rows may be clipped/covered although both observed
        # fit edges are clear. Fitting/foot/margin/white tests remain.
        fractions=([r['y'] for r in path['current_fit_edge_rows']]
                   if path.get('flowing') and path.get('current_fit_edge_rows')
                   and path['near_clearance']>.35
                   and abs(14*(path['centers'][1]-w/2)/(w/2)+
                           3*(path['centers'][1]-path['centers'][0])/(w/2))<7
                   else (.93,.87,.83,.77,.70))
        for fraction in fractions:
            y = min(h-1, int(fraction*h))
            row = (mask[max(0,y-2):min(h,y+3)] == 1).sum(0) > 2
            xs = np.flatnonzero(row)
            if len(xs) < .18*w or xs[0] < 3 or xs[-1] > w-4:
                continue
            valid = []
            for a,b in ((xs[0]+5,xs[0]+17),(xs[-1]-16,xs[-1]-4)):
                strip = white[max(0,y-2):min(h,y+3), a:b]
                support = mask[max(0,y-2):min(h,y+3), a:b] == 1
                valid.append(bool(strip.size and strip.mean() < .45 and support.mean() > .65))
            paired += all(valid)
        path['paired_current_visible_edges'] = int(paired)
        if paired < 2 or path['near_clearance'] < .30:
            return None, 'edges_hidden_or_small_margin'
        if abs(path['target']-w/2) > .18*w:
            return None, 'curved_or_off_center_corridor'
        return path, 'current_edges_verified'

    def apply(self, base, mask, raw, detections, status, *, dt_s, fresh,
              yaw_rate, traffic_context):
        dt = float(np.clip(dt_s, .01, .25))
        self.elapsed += dt
        h, w = mask.shape
        self.memory.advance(status,dt,fresh)
        rh, rw = raw.shape[:2]
        fountains = [d for d in detections if d['class'] == 'fountain'
                     and d['confidence'] >= .30
                     and (d['box'][2]-d['box'][0]) >= .025*rw
                     and (d['box'][3]-d['box'][1]) >= .04*rh]
        while self.hits and self.elapsed-self.hits[0] > .5:
            self.hits.popleft()
        if fountains:
            self.hits.append(self.elapsed)
            if len(self.hits) >= 2:
                self.last_cue = self.elapsed
                if any(d['box'][3]>.55*rh and max((d['box'][2]-d['box'][0])/rw,
                       (d['box'][3]-d['box'][1])/rh)>.12 for d in fountains):
                    self.last_close_cue = self.elapsed
        cue_age = self.elapsed-self.last_cue
        active = cue_age <= 1.2
        self.info = dict(reason='no_confirmed_fountain', detected=bool(fountains),
                         detections=fountains, confirmed=active,
                         cue_age_s=round(cue_age,3) if cue_age < 100 else None,
                         owns_path=False, base_mode=base.mode, base_speed=base.speed)
        if not active:
            self.observations.clear()
            self.braking = False
            self.memory.clear()
            return base
        # A water policy does not choose/complete a junction branch. Those
        # decisions still belong to Traffic's selected legal corridor.
        turn_or_junction = (traffic_context.get('turn_direction')
            or traffic_context.get('crossing') or traffic_context.get('junction_mouth_open')
            or base.junction_suspect or base.junction_hold_path)
        hard = (not fresh or base.mode in ('manual_assist','status_stale','final_status_stale')
            or turn_or_junction or base.edge_trend_guard or base.boundary_recovery_guard
            or base.geometry_guard or abs(yaw_rate) >= 18)
        if hard:
            self.info['reason'] = 'stale_turn_or_observed_boundary_veto'
            self.memory.clear()
            return base
        if base.preview_bend>.25 and base.confidence==1 and base.preview_quality>=.85:
            # Current far bend belongs to Road; a straight fitted near stem
            # must not raise its speed or replace its bend steering.
            self.info['reason']='current_observed_bend_road_priority'
            self.memory.clear()
            return base
        if base.mode=='corner_momentum_brake':
            self.info['reason']='current_observed_road_speed_priority'
            return base
        path, reason = self.current_path(mask, raw)
        memory_used=False
        if path is None and reason in ('edges_hidden_or_small_margin','no_current_connected_corridor'):
            path=self.memory.current_support(mask,raw,status,self.elapsed)
            memory_used=path is not None
            if memory_used:reason='short_memory_with_current_visible_edge'
        self.info.update(reason=reason, current_path=path,
                         short_memory=self.memory.info,memory_used=memory_used)
        while self.observations and self.elapsed-self.observations[0][0] > .45:
            self.observations.popleft()
        if path is None:
            # Short missing measurements may retain agreement, never permission.
            return base
        target = float(path['target'])
        if self.observations and abs(target-self.observations[-1][1]) >= .05*w:
            self.observations.clear()
        self.observations.append((self.elapsed, target))
        agreed = memory_used or (len(self.observations) >= 2 and self.elapsed-self.observations[0][0] >= .06)
        wet = base.occlusion_fraction >= .06 or base.spray_prealert or base.mode.startswith('occlusion_')
        self.info.update(reason='prearmed_current_corridor' if not wet else 'confirm_current_corridor',
                         agreement_frames=len(self.observations), wet=wet)
        # Only current TWO-edge fitted geometry can renew the snapshot.
        # A one-edge continuation never extends its own age/distance budget.
        if (agreed and not memory_used and path['flowing']
                and path['near_clearance']>.35 and abs(yaw_rate)<8):
            self.memory.arm(path,status,self.elapsed)
        prepare = (not wet and self.elapsed-self.last_close_cue <= .7
            and base.confidence==1 and base.preview_quality>=.85
            and not base.steering_settle_guard and not base.mask_shift_guard)
        if not agreed or (not wet and not prepare):
            return base
        # Bounded target smoothing is allowed only if the CURRENT path agrees.
        # Do not freeze a previous large steering angle or use a hidden path.
        desired = float(np.clip(20*(target-w/2)/(w/2)-.08*yaw_rate, -12, 12))
        if path['flowing']:
            near, mid, far = path['centers']
            desired = float(np.clip(14*(mid-w/2)/(w/2)+3*(mid-near)/(w/2)-.06*yaw_rate, -8, 8))
        angle = desired
        if self.last_angle is not None and self.elapsed-self.last_angle_time <= .20 and abs(desired-self.last_angle) < 5:
            angle = float(np.clip(desired, self.last_angle-22*dt, self.last_angle+22*dt))
        actual_speed = float(status.get('Speed',0))
        actual_wheel = status.get('Angle')
        flowing = (path['flowing'] and path['near_clearance'] > .35
                   and abs(yaw_rate) < 10 and abs(desired) < 7)
        # The fast band also requires a clear long preview. A locally straight
        # near corridor is not enough when the next bend is missing/ambiguous.
        fast_flowing = (flowing and not memory_used
            and path.get('source') == 'current_fitted_fountain_edges'
            and path.get('paired_current_visible_edges', 0) >= 2
            and not path.get('missing_edge_estimated')
            and base.preview_quality >= .85 and abs(base.preview_bend) < .18
            and not base.clipped_near_preview)
        request = band(34.0) if fast_flowing else band(28.0) if flowing and not memory_used else 22.0 if flowing else 10.0
        ceiling = band(38.0) if fast_flowing else band(31.0) if flowing and not memory_used else 25.0 if flowing else 13.0
        release = band(35.0) if fast_flowing else band(27.0) if flowing and not memory_used else 21.0 if flowing else 10.0
        self.braking = actual_speed > ceiling or (self.braking and actual_speed > release)
        aligned = actual_wheel is not None and abs(float(actual_wheel)-desired) <= 6
        if self.braking or not aligned:
            self.info.update(reason='measured_overspeed_brake' if self.braking else 'physical_wheel_alignment',
                             ceiling=ceiling, release=release, actual_speed=actual_speed, desired_angle=desired)
            # Real momentum/steering mismatch remains a brake, not a motion floor.
            request = 0.0
        else:
            self.info['reason'] = ('short_memory_current_edge_pass' if memory_used else 'current_fountain_prepare' if prepare else 'current_fountain_corridor_pass')
        self.last_angle, self.last_angle_time = angle, self.elapsed
        centers = path.get('centers', (path['center'], target, None))
        widths = path.get('widths', (path['width'], path['target_width'], None))
        self.info.update(owns_path=True, request=request, desired_angle=desired,
                         actual_wheel=actual_wheel, ceiling=ceiling)
        return replace(base, speed=request, angle=angle, target_x=target,
                       mode=('fountain_prepare' if prepare else 'fountain_visible_pass') if request > 0 else 'fountain_physical_brake',
                       confidence=2/3, centers=centers, widths=widths,
                       near_left=float(path['near_bounds'][0]), near_right=float(path['near_bounds'][1]),
                       near_clearance=path['near_clearance'], current_path={**path, 'current_fast_preview_verified': bool(fast_flowing)},
                       occlusion_hold=False, spray_prealert=False, preview_quality=.6)
