from speed_profile import band
"""Online sign/vehicle policy using image geometry and recent observations only."""
from collections import deque
from turn_path_contract import TurnPathContract
from turn_alignment_request import TurnAlignmentRequest
from outgoing_path_contract import current_outgoing_contract
from sign_brake_revalidation import SignBrakeRevalidation
from sign_approach_reserve import SignApproachReserve
from dataclasses import replace

import cv2
import numpy as np
from junction_path import current_junction_floor, current_legal_branch_arc
from obstacle_path import observed_gap_corridor, close_box_off_current_corridor, close_box_off_current_near_curve
from near_motion_relevance import remote_box_outside_current_near_motion, current_near_motion_evidence

SIGN_CLASSES = frozenset(('turn_left', 'turn_right', 'go_straight',
                          'no_left_turn', 'no_right_turn',
                          'turn_prohibition_unknown'))


def conflicting_sign_faces(detections):
    """Competing classes on one visible face cannot establish a turn vote."""
    signs = [d for d in detections if d['class'] in SIGN_CLASSES
             and d['confidence'] >= .45
             and min(d['box'][2]-d['box'][0], d['box'][3]-d['box'][1]) >= 10]
    conflicts = []
    for i, first in enumerate(signs):
        for second in signs[i+1:]:
            if first['class'] == second['class']:
                continue
            a, b = first['box'], second['box']
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(
                0, min(a[3], b[3])-max(a[1], b[1]))
            area_a = max(1, (a[2]-a[0])*(a[3]-a[1]))
            area_b = max(1, (b[2]-b[0])*(b[3]-b[1]))
            if intersection / (area_a+area_b-intersection) >= .35:
                conflicts.append([first['class'], second['class']])
    return conflicts


class TrafficController:
    def __init__(self):
        self.votes = deque()
        self.elapsed = 0.0
        self.pending_sign = None
        self.sign_seen_at = -1e9
        self.sign_size = 0.0
        self.sign_brake_limit = None
        self.sign_brake_revalidation=SignBrakeRevalidation()
        self.sign_approach_reserve=SignApproachReserve()
        self.turn_direction = 0
        self.turn_wheel_ready = False
        self.turn_entry_braking = False
        self.turn_progress = 0.0
        self.turn_elapsed = 0.0
        self.exit_count = 0
        self.junction_mouth_count = 0
        self.t_junction_count = 0
        self.legal_side_count = 0
        self.cooldown = 0.0
        self.handover_remaining = 0.0
        self.avoid_side = None
        self.avoid_clear_s = 0.0
        self.side_probe_distance = 0.0
        self.side_probe_motion_s = 0.0
        self.last_angle = 0.0
        self.info = {}
        self.turn_contract = TurnPathContract()
        self.turn_alignment_request = TurnAlignmentRequest()
        self.vehicle_growth = deque()
        self.gap_clear_s = 0.0
        self.previous_gap_target = None
        self.depth_recovery_stopped_s = 0.0
        self.depth_recovery_target = None
        self.depth_recovery_active = False

    @staticmethod
    def road_bounds(mask, y):
        h, w = mask.shape
        y = int(np.clip(y, 0, h-1))
        band = mask[max(0,y-2):min(h,y+3)] == 1
        row = band.sum(0) > len(band)//2
        xs = np.flatnonzero(row)
        if len(xs) < 8:
            return None
        return float(xs[0]), float(xs[-1])

    @classmethod
    def legal_side_opening(cls, route_mask, direction):
        """Current one-sided branch beyond the observed incoming perspective.

        A normal widening lane is insufficient. Both preview rows must
        reach the requested side AND extend beyond a fit of the farther
        incoming lane. Never infer the branch from remembered scene data.
        """
        h,w=route_mask.shape
        points=[]
        for fraction in (.45,.50,.55,.60):
            bounds=cls.road_bounds(route_mask,fraction*h)
            if bounds and bounds[0]>.03*w and bounds[1]<.97*w:
                points.append((fraction,*bounds))
        if len(points)<3:return None
        rows=np.array(points,float)
        coeff=np.linalg.lstsq(np.c_[rows[:,0],np.ones(len(rows))],rows[:,1:],rcond=None)[0]
        residual=np.abs(np.c_[rows[:,0],np.ones(len(rows))]@coeff-rows[:,1:]).max()
        # Curved/occluded far edges cannot define a clean incoming stem.
        if residual>.025*w:return None
        edge=1 if direction>0 else 0
        observations=[]
        for fraction in (.70,.75,.80):
            bounds=cls.road_bounds(route_mask,fraction*h)
            if bounds is None:return None
            expected=float(np.array([fraction,1.0])@coeff[:,edge])
            extension=direction*(bounds[edge]-expected)
            saturated=(bounds[1]>.98*w if direction>0 else bounds[0]<.02*w)
            observations.append(dict(y=fraction,bounds=list(bounds),extension_px=float(extension),side_visible=saturated and extension>.08*w))
        near=cls.road_bounds(route_mask,.85*h)
        # The candidate branch must have reached the close region, while
        # the optical-axis approach still has observed current ground.
        foot=route_mask[int(.88*h):int(.97*h),int(.43*w):int(.57*w)]
        close=bool(near and (near[1]>.98*w if direction>0 else near[0]<.02*w))
        if not close or foot.size==0 or foot.mean()<.90:return None
        if sum(x['side_visible'] for x in observations)<2:return None
        return dict(source='current_legal_side_opening',direction=direction,
                    incoming_fit_error_px=float(residual),rows=observations,
                    near_bounds=list(near),foot_road_fraction=float(foot.mean()))

    def update(self, detections, mask, base, *, raw_shape, dt_s, speed, yaw_rate, actual_angle=None, raw_image=None):
        self.turn_alignment_request.begin_observation()
        dt = float(np.clip(dt_s, .01, .25))
        self.elapsed += dt
        self.cooldown = max(0.0, self.cooldown-dt)
        self.handover_remaining=max(0.0,self.handover_remaining-dt)
        rh, rw = raw_shape[:2]
        h, w = mask.shape
        sx, sy = w/rw, h/rh
        path=base.current_path
        depth_scope=bool(base.mode=='near_curve_tracking' and path is not None
            and path.get('source')=='current_connected_near_curve'
            and not self.turn_direction and self.pending_sign is None
            and self.handover_remaining<=0 and raw_image is not None
            and actual_angle is not None and np.isfinite([speed,actual_angle,base.angle]).all())
        if not depth_scope:
            self.depth_recovery_stopped_s=0.0
            self.depth_recovery_target=None
            self.depth_recovery_active=False
        else:
            target=float(path['target'])
            agrees=(self.depth_recovery_target is not None
                and abs(target-self.depth_recovery_target)<.06*w)
            stalled=(speed<.5 and base.speed>0 and
                abs(float(actual_angle)-base.angle)<=5 and abs(yaw_rate)<5
                and self.info.get('mode')=='occluded_traffic_stop'
                and self.info.get('traffic_cap')==0)
            self.depth_recovery_stopped_s=(self.depth_recovery_stopped_s+dt
                if stalled and agrees else 0.0)
            self.depth_recovery_target=target
        allow_depth_recovery=(self.depth_recovery_active or
            self.depth_recovery_stopped_s>=.60)
        # Use only the road component attached to the current vehicle region.
        road = cv2.morphologyEx((mask==1).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((3,5),np.uint8))
        n, labels = cv2.connectedComponents(road)
        counts = np.bincount(labels[int(.9*h):,int(.28*w):int(.72*w)].ravel(),minlength=n)
        counts[0] = 0
        route_mask = (labels==int(np.argmax(counts))).astype(np.uint8) if counts.max()>25 else np.zeros_like(mask)
        signs=[]
        vehicles=[]
        uncertain_side_contacts=0
        vehicle_contact_evidence=[]
        for d in detections:
            x1,y1,x2,y2=d['box']
            size=min(x2-x1,y2-y1)
            if d['class'] in SIGN_CLASSES:
                if d['confidence'] >= .45 and size >= 10:
                    unclipped = x1>2 and y1>2 and x2<rw-2 and y2<rh-2
                    signs.append((size,d['confidence'],d['class'],unclipped))
            elif d['class'] == 'vehicle':
                # Only current road contact, not scenery or floating vehicles.
                bounds=self.road_bounds(route_mask,(y2+rh*.025)*sy)
                if bounds is None or y2 < .43*rh:
                    continue
                a,b=bounds
                left,right=x1*sx,x2*sx
                overlap=max(0,min(b,right)-max(a,left))
                if overlap/max(1,right-left) < .35:
                    continue
                # A box can overlap the road while its actual ground anchor
                # is outside it. Test the central foot on CURRENT connected
                # road pixels, not the hull of all pixels at one scanline.
                # An immediate box across the optical axis is still a stop
                # hazard when its road contact is hidden by the object.
                cx=(left+right)/2
                radius=max(2,.25*(right-left))
                ya=min(h-1,max(0,int((y2+.01*rh)*sy)))
                yb=min(h,max(ya+1,int((y2+.045*rh)*sy)))
                xa=max(0,int(cx-radius)); xb=min(w,max(xa+1,int(cx+radius)+1))
                contact=route_mask[ya:yb,xa:xb]
                supported=bool(contact.size and contact.mean()>=.60)
                across_axis=left<=w/2<=right
                vehicle_contact_evidence.append(dict(left=left,right=right,
                    bottom=y2/rh,central_road_fraction=float(contact.mean()) if contact.size else 0,
                    supported=supported,across_axis=across_axis))
                corridor=close_box_off_current_corridor(mask,raw_image,d['box'],base,
                    actual_angle=actual_angle,allow_early=(not self.turn_direction and
                    self.pending_sign in (None,'go_straight')))
                if corridor is None:
                    corridor=close_box_off_current_near_curve(mask,raw_image,d['box'],base)
                if (corridor is None and self.turn_direction==0 and self.pending_sign is None
                    and abs(self.last_angle-base.angle)<=5
                    and not any(q['class'] in SIGN_CLASSES and q['confidence']>=.45
                        and min(q['box'][2]-q['box'][0],q['box'][3]-q['box'][1])>=10
                        for q in detections)):
                    corridor=remote_box_outside_current_near_motion(mask,raw_image,d['box'],base,
                        wheel=actual_angle,speed=speed,
                        allow_depth_recovery=allow_depth_recovery)
                if corridor is not None and corridor.get('depth_recovery_only'):
                    self.depth_recovery_active=True
                vehicle_contact_evidence[-1]['off_current_corridor']=corridor
                if corridor is not None:
                    # This box is irrelevant to the CURRENT observed Road
                    # path. Keep all Road/Fountain/Recovery speed limits.
                    continue
                if not supported and not across_axis:
                    uncertain_side_contacts+=1
                    continue
                vehicles.append(dict(left=left,right=right,bottom=y2/rh,
                                     width=(x2-x1)/rw,road_left=a,road_right=b))
        visible = max(signs, default=None)
        sign_conflicts = conflicting_sign_faces(detections)
        if sign_conflicts:
            self.votes.clear()
        if visible is not None and not sign_conflicts:
            size,confidence,kind,unclipped=visible
            if self.pending_sign is None or kind == self.pending_sign:
                self.sign_seen_at=self.elapsed
            # Tiny detections are slowdown hints, not turn commands.
            if size >= 20 and confidence >= .55 and unclipped:
                self.votes.append((self.elapsed,kind,confidence,size))
                if kind == self.pending_sign:
                    self.sign_size=max(self.sign_size,size)
        while self.votes and self.elapsed-self.votes[0][0]>.6:
            self.votes.popleft()
        if self.turn_direction==0 and self.cooldown==0 and self.votes and self.pending_sign is None:
            counts={k:sum(v[1]==k for v in self.votes) for _,k,_,_ in self.votes}
            chosen=max(counts,key=counts.get)
            if counts[chosen]>=3 and counts[chosen]/len(self.votes)>=.75:
                if self.pending_sign != chosen:
                    self.sign_size=0.0
                self.pending_sign=chosen
                self.sign_size=max(self.sign_size,max(v[3] for v in self.votes if v[1]==chosen))
        sign_hint=visible is not None or self.pending_sign is not None
        # A restriction must survive the slow approach after its face leaves
        # the camera. It applies to the next junction, then expires or clears
        # with the maneuver; it is never a stored map location.
        restrictions=('no_left_turn','no_right_turn','turn_prohibition_unknown')
        sign_expiry=12.0 if self.pending_sign in restrictions else 3.0
        if self.pending_sign and self.elapsed-self.sign_seen_at>sign_expiry and self.turn_direction==0:
            self.pending_sign=None
            self.sign_size=0.0
            self.votes.clear()
        b70=self.road_bounds(route_mask,.70*h)
        b85=self.road_bounds(route_mask,.85*h)
        # Widening farther ahead is a preview. Wait for the opening to reach
        # the near camera region before steering across the inner corner.
        crossing=bool(b70 and b70[1]-b70[0]>.78*w)
        mouth_open=bool(crossing and b85 and b85[1]-b85[0]>.92*w and
                        b85[0]<.04*w and b85[1]>.96*w)
        self.junction_mouth_count=self.junction_mouth_count+1 if mouth_open else 0
        # A no-right sign permits straight travel at a crossroad. Choose left
        # only when the current connected road shows a transverse opening
        # and no forward continuation at the middle preview row.
        middle=route_mask[max(0,int(.55*h)-2):min(h,int(.55*h)+3),
                          int(.43*w):int(.57*w)]
        branch=route_mask[max(0,int(.70*h)-2):min(h,int(.70*h)+3)]
        foot=route_mask[int(.88*h):int(.97*h),int(.43*w):int(.57*w)]
        t_shape=bool(mouth_open and middle.size and middle.mean()<.15 and
                     foot.size and foot.mean()>.85 and base.confidence>.6 and
                     base.near_clearance>.28 and
                     branch[:,int(.08*w):int(.32*w)].mean()>.75 and
                     branch[:,int(.68*w):int(.92*w)].mean()>.75 and
                     branch[:,int(.43*w):int(.57*w)].mean()>.75 and
                     base.occlusion_fraction<.12)
        self.t_junction_count=self.t_junction_count+1 if t_shape else 0
        requested_turn=(self.pending_sign if self.pending_sign in ('turn_left','turn_right')
                        else 'turn_left' if self.pending_sign=='no_right_turn' and
                        self.t_junction_count>=2 else 'turn_right' if
                        self.pending_sign=='no_left_turn' and self.t_junction_count>=2 else None)
        requested_direction=(1 if requested_turn=='turn_right' else -1 if requested_turn=='turn_left' else 0)
        side_evidence=(self.legal_side_opening(route_mask,requested_direction)
            if requested_direction and base.occlusion_fraction<.12 and base.confidence>=.6 else None)
        self.legal_side_count=self.legal_side_count+1 if side_evidence is not None else 0
        # Select the observed legal branch regardless of entry momentum.
        # The physical-wheel/speed alignment phase brakes before advancing.
        if (self.turn_direction==0 and requested_turn is not None
                and not sign_conflicts
                and (self.junction_mouth_count>=2 or self.legal_side_count>=2) and self.sign_size>=25):
            self.turn_direction=1 if requested_turn=='turn_right' else -1
            self.turn_wheel_ready=False
            self.turn_entry_braking=speed>14.0
            self.turn_progress=0
            self.turn_elapsed=0
            self.exit_count=0
        decision=base
        mode='road'
        goal_angle=base.angle
        cap=base.speed
        stages=[dict(layer='road',cap=float(cap),mode=base.mode)]
        branch_owner=False
        branch_reobserve=False
        incoming_near_proof=None
        approach_limit=None
        size_only_close_preview=False
        flow_approach_resolved=False
        reserve_limit=self.sign_approach_reserve.update(detections,
            shape=raw_shape,route=self.pending_sign,turning=bool(self.turn_direction),
            speed=float(speed),wheel=actual_angle,yaw=float(yaw_rate),dt=dt,
            fresh=not sign_conflicts and base.mode not in
                ('status_stale','final_status_stale','manual_assist'))
        if sign_hint:
            # A far sign identifies the route. Only current resolved road
            # permits the higher approach cap; close/conflict braking below
            # remains authoritative, with actual speed and wheel feedback.
            far_resolved=(base.confidence==1 and base.preview_quality>=.85
                and base.near_clearance>.34 and abs(base.angle)<8
                and abs(yaw_rate)<12 and base.occlusion_fraction<.025
                and not base.junction_suspect and not crossing and not mouth_open
                and not base.geometry_guard and not base.edge_trend_guard
                and not base.boundary_recovery_guard and not base.steering_settle_guard)
            # Fountain's owned path already verifies CURRENT paired edges.
            # Its 2/3 scanline ratio is not an uncertainty probability. Reuse
            # this same geometry for approach; base physical brake still
            # caps speed, and actual junction/conflict/obstacle gates follow.
            owned=base.current_path or {}
            # Reuse the producing layer's CURRENT fast-preview verification.
            # Old-band limits must not downgrade its verified 28/32 band solely
            # because a sign is visible. Base and close-turn hazard caps remain.
            owned_fast=bool(owned.get('current_fast_preview_verified'))
            flow_request_limit=band(34.0) if owned_fast else band(28.0)
            flow_actual_limit=band(38.0) if owned_fast else band(31.0)
            flow_approach_resolved=bool(
                base.mode in ('fountain_visible_pass','fountain_prepare')
                and 0<base.speed<=flow_request_limit and speed<=flow_actual_limit
                and owned.get('source')=='current_fitted_fountain_edges'
                and owned.get('flowing') and not owned.get('missing_edge_estimated')
                and owned.get('paired_current_visible_edges',0)>=2
                and base.near_clearance>.35 and abs(base.angle)<7 and abs(yaw_rate)<8
                and actual_angle is not None and abs(float(actual_angle)-base.angle)<=5
                and not self.turn_direction and not sign_conflicts
                and not base.junction_suspect and not crossing and not mouth_open
                and not base.junction_hold_path and not base.geometry_guard
                and not base.edge_trend_guard and not base.boundary_recovery_guard
                and not base.steering_settle_guard and not base.mask_shift_guard)
            far_resolved=far_resolved or flow_approach_resolved
            # A clear straight road does not resolve a tiny/uncertain sign.
            # Keep fast approach only for a voted straight-through route.
            # Actual cap braking below still enforces this rolling ceiling.
            # A tiny unclipped sign is a distant warning, not yet a route
            # choice. Do not force clear straight road to20 solely because
            # this hint exists. Vote/branch/close-face and real braking remain.
            distant_hint=(far_resolved and self.pending_sign is None
                and visible is not None and 10<=visible[0]<20
                and visible[1]>=.55 and visible[3] and not sign_conflicts)
            size_only_close_preview=bool(self.pending_sign in ('turn_left','turn_right')+restrictions
                and not self.turn_direction and self.sign_size>=30 and far_resolved
                and base.preview_bend < .25
                and not crossing and not mouth_open and not sign_conflicts
                and actual_angle is not None
                and abs(float(actual_angle)-base.angle)<=4)
            # Required turns need braking reserve before the mouth opens.
            # A clear incoming lane does not remove measured wheel/brake lag.
            # Distant approach and straight/prohibition signs retain their bands.
            approach_limit=(18.0 if size_only_close_preview and self.pending_sign in ('turn_left','turn_right')
                            else band(24.0) if size_only_close_preview
                            else band(36.0) if far_resolved and self.pending_sign=='go_straight'
                            else band(32.0) if distant_hint or (far_resolved and self.pending_sign in
                                ('turn_left','turn_right') and self.sign_size<30)
                            else band(20.0) if far_resolved else 18.0)
            cap=min(cap,approach_limit)
            mode='sign_approach'
            # BTC's positive speed command is not a measured speed target.
            # Withdraw torque before entering a confirmed tight turn.
            if self.pending_sign in ('turn_left','turn_right')+restrictions:
                close_turn=(self.sign_size>=30 or crossing or mouth_open)
                # A large sign alone is not the junction mouth. Carry the
                # current resolved incoming lane while the PHYSICAL wheel
                # follows it; the existing mouth/curve caps stay authoritative.
                close_limit=band(24.0) if size_only_close_preview and self.pending_sign in restrictions else 18.0
                approach_limit=min(approach_limit,close_limit if close_turn else (36.0 if far_resolved else 20.0))
                cap=min(cap,approach_limit)
                if close_turn and speed>close_limit+3.0:
                    cap=0.0
                    mode='sign_brake'
            # Current optical growth may show insufficient braking reserve
            # before a face reaches the old large-sign threshold. Activate
            # the existing lower ceiling only; no route/steering permission.
            if reserve_limit is not None:
                approach_limit=min(approach_limit,reserve_limit)
                cap=min(cap,approach_limit)
            if self.pending_sign=='turn_prohibition_unknown' and self.t_junction_count>=2:
                cap=0.0
                mode='restriction_direction_unknown_stop'
            if sign_conflicts and not self.turn_direction:
                cap=min(cap,6.0)
                if speed>8.0 or self.junction_mouth_count>=2:
                    cap=0.0
                mode='sign_conflict_wait'
        # A positive low request does not brake entry momentum. Enforce the
        # selected approach limit with measured speed, retaining the brake
        # target across short clear/uncertain preview alternations. Release
        # above zero speed once actual speed reaches the lower exit band.
        # A retained brake belongs to the path that needed it. Recheck it
        # against the CURRENT observed legal continuation before forcing
        # the car to the old18km/h band despite a newly resolved road.
        revalidate_eligible=bool(sign_hint
            and self.pending_sign in ('go_straight','no_left_turn','no_right_turn')
            and (far_resolved if sign_hint else False)
            and base.mode in ('normal','fountain_prepare','fountain_visible_pass')
            and not self.turn_direction and not sign_conflicts and not vehicles
            and not uncertain_side_contacts and not crossing and not mouth_open
            and not t_shape and not base.junction_suspect and not base.junction_hold_path
            and not base.geometry_guard and not base.edge_trend_guard
            and not base.boundary_recovery_guard and not base.steering_settle_guard
            and not base.mask_shift_guard and base.near_clearance>.35
            and abs(base.angle)<6
            and (base.occlusion_fraction<.025 or flow_approach_resolved))
        self.sign_brake_limit=self.sign_brake_revalidation.evaluate(
            self.sign_brake_limit,limit=approach_limit,request=base.speed,
            speed=speed,wheel=actual_angle,angle=base.angle,yaw=yaw_rate,dt=dt,
            route=self.pending_sign,eligible=revalidate_eligible,
            path_source=(base.current_path or {}).get('source','current_resolved_road'))
        if self.sign_brake_limit is not None:
            if speed<=self.sign_brake_limit+1.5:
                self.sign_brake_limit=None
        if approach_limit is not None and speed>approach_limit+3.0:
            self.sign_brake_limit=(approach_limit if self.sign_brake_limit is None
                                   else min(self.sign_brake_limit,approach_limit))
        if self.sign_brake_limit is not None:
            cap=0.0
            mode='sign_speed_brake'
        stages.append(dict(layer='sign',cap=float(cap),mode=mode))
        if self.turn_direction:
            if self.turn_entry_braking and speed<=14.0:
                self.turn_entry_braking=False
            if not self.turn_entry_braking and (actual_angle is None or (
                    self.turn_direction*float(actual_angle)>=18 and speed<=14)):
                self.turn_wheel_ready=True
            # Waiting/Manual stillness is not time spent traversing a turn.
            if speed>1.0:
                self.turn_elapsed+=dt
            self.turn_progress+=self.turn_direction*float(yaw_rate)*dt
            progress=max(0,self.turn_progress)
            turn_angle=self.turn_direction*float(np.interp(progress,[0,45,70,90],[25,23,12,0]))
            blend=float(np.clip((progress-65)/20,0,1))
            goal_angle=(1-blend)*turn_angle+blend*base.angle
            # The baseline's settle check targets the center/straight route,
            # which differs from this committed branch. Use our own bounded
            # turn request while preserving the perception/edge veto below.
            cap=band(20.0) if speed<=band(24) else 0.0
            if abs(base.angle-goal_angle)>30:
                cap=0.0 if speed>6 else min(cap,3.0)
            mode='turn_right' if self.turn_direction>0 else 'turn_left'
            if not self.turn_wheel_ready:
                # Wait for the PHYSICAL wheel, not the intended command.
                # Positive small throttle merely coasts above its target;
                # request zero here to arrest entry momentum before turning.
                cap=3.0 if (speed<=4 and base.confidence==1 and base.near_clearance>.30 and base.occlusion_fraction<.12 and not sign_conflicts and foot.size and foot.mean()>.95) else 0.0
                mode='turn_prepare_creep' if cap else 'turn_align'
                if self.turn_entry_braking:
                    # Brake on the currently observed incoming lane before
                    # commanding a large turn while entry momentum persists.
                    # Current road steering can still correct that approach.
                    cap=0.0
                    goal_angle=base.angle
                    mode='turn_entry_brake'
            elif base.near_clearance<.20 and speed>6:
                cap=0.0
                mode='turn_edge_brake'
            widths=base.widths
            centers=base.centers
            perspective=(all(v is not None for v in widths+centers) and
                widths[0]>1.10*widths[1] and widths[1]>1.25*widths[2] and
                widths[2]>.08*w and widths[2]<.70*widths[1] and
                centers[1]-widths[1]/2>.03*w and
                centers[1]+widths[1]/2<.97*w)
            # A valid outgoing lane can be curved or meet at a non90-degree
            # angle. Use current road perspective rather than a fixed yaw goal.
            aligned=(progress>35 and base.confidence==1 and perspective and
                     not base.junction_suspect and not base.junction_hold_path and
                     base.near_clearance>.28 and abs(base.angle)<6)
            self.exit_count=self.exit_count+1 if aligned else 0
            if self.exit_count>=3:
                self.turn_direction=0
                self.pending_sign=None
                self.sign_size=0.0
                self.votes.clear()
                self.cooldown=1.0
                self.handover_remaining=.60
                goal_angle=base.angle
                cap=min(base.speed,band(20.0))
                mode='turn_exit'
            elif self.turn_elapsed>8 or progress>115:
                cap=0
                mode='turn_reobserve'
                branch_reobserve=True
        elif self.handover_remaining>0:
            cap=min(cap,band(20.0))
            mode='turn_handover'
        stages.append(dict(layer='turn',cap=float(cap),mode=mode))
        committed_goal_angle=goal_angle
        obstacle= max(vehicles,key=lambda v:v['bottom'],default=None)
        gap_target=None
        remote_vehicle=False
        pass_evidence=None
        self.gap_clear_s=max(0.0,self.gap_clear_s-dt)
        if obstacle is not None:
            self.avoid_clear_s=0
            a,b=obstacle['road_left'],obstacle['road_right']
            margin=max(.010*w,.08*(b-a))
            blocks_ego = obstacle['left']-margin <= w/2 <= obstacle['right']+margin
            gaps={'left':(a+margin,obstacle['left']-margin),
                  'right':(obstacle['right']+margin,b-margin)}
            gaps={side:gap for side,gap in gaps.items() if gap[1]-gap[0]>=max(.035*w,.27*(b-a))}
            cap=min(cap,14.0 if obstacle['bottom']<.65 else 10.0)
            mode='vehicle_approach'
            # Small distant objects are a preview, not a near avoidance target.
            # Only retain road steering when current full road evidence is
            # resolved, and growth does not imply rapid approach.
            self.vehicle_growth.append((self.elapsed,obstacle['width']))
            while self.vehicle_growth and self.elapsed-self.vehicle_growth[0][0]>.5:
                self.vehicle_growth.popleft()
            growth=0.0
            if len(self.vehicle_growth)>1:
                age=self.elapsed-self.vehicle_growth[0][0]
                growth=(obstacle['width']-self.vehicle_growth[0][1])/max(age,.01)
            remote_vehicle=(obstacle['width']<.055 and obstacle['bottom']<.58
                and growth<.06 and base.confidence==1 and base.preview_quality>=.8
                and base.near_clearance>.35 and abs(base.angle)<6 and abs(yaw_rate)<8
                and not base.junction_suspect and not crossing
                and self.pending_sign is None and not sign_hint
                and base.occlusion_fraction<.025 and not self.turn_direction)
            if remote_vehicle:
                cap=min(base.speed,band(30.0))
                goal_angle=base.angle
                mode='vehicle_remote_preview'
                self.avoid_side=None
                if speed>cap+3:cap=0.0
            elif not self.turn_direction:
                if self.avoid_side not in gaps:
                    self.avoid_side=max(gaps,key=lambda s:gaps[s][1]-gaps[s][0],default=None)
                if self.avoid_side is not None:
                    lo,hi=gaps[self.avoid_side]
                    gap_target=(lo+hi)/2
                    goal_angle=16*((gap_target-w/2)/(w/2))-.06*yaw_rate
                    mode='avoid_'+self.avoid_side
                    # Increase only this existing avoidance request, not road
                    # speed or turning. Observe a continuous current strip to
                    # the chosen gap; never use a remembered unobserved lane.
                    path=observed_gap_corridor(route_mask,raw_image,(lo,hi),obstacle['bottom'])
                    resolved=(path is not None and base.speed>0 and base.mode not in ('status_stale','manual_assist','final_status_stale') and base.confidence==1
                        and base.preview_quality>=.8 and base.near_clearance>.28
                        and base.occlusion_fraction<.025 and not sign_hint
                        and self.pending_sign is None and not sign_conflicts
                        and not crossing and not mouth_open and not base.junction_suspect
                        and abs(yaw_rate)<12 and abs(goal_angle)<12
                        and actual_angle is not None and abs(float(actual_angle)-goal_angle)<=4
                        and not base.geometry_guard and not base.edge_trend_guard
                        and not base.boundary_recovery_guard)
                    agrees=(self.previous_gap_target is not None
                            and abs(gap_target-self.previous_gap_target)<.04*w)
                    self.gap_clear_s=self.gap_clear_s+2*dt if resolved and agrees else 0.0
                    self.previous_gap_target=gap_target if resolved else None
                    if self.gap_clear_s>=.20:
                        cap=min(base.speed,band(32.0) if obstacle['bottom']<.65 else band(26.0))
                        if speed>cap+3:
                            cap=0.0
                            mode='vehicle_resolved_gap_brake'
                    pass_evidence=dict(eligible=bool(resolved),agreement_s=self.gap_clear_s,
                        speed_request=float(cap),current_path=path)
                elif obstacle['bottom']>.58:
                    cap=0
                    mode='vehicle_no_gap_stop'
                    # A distant scanline can have too little perspective gap
                    # while the visible corridor immediately ahead is clear.
                    # One bounded advance may reveal a valid gap; never drive
                    # into a center obstacle or repeatedly renew this budget.
                    immediate=route_mask[int(.85*h):,int(.46*w):int(.54*w)]
                    side_clear=min(abs(obstacle['left']-w/2),abs(obstacle['right']-w/2))>.06*w
                    near_safe=(not blocks_ego and side_clear and
                        obstacle['bottom']<.80 and immediate.size and
                        immediate.mean()>.85 and base.confidence==1 and
                        base.near_clearance>.28 and abs(base.angle)<8 and
                        not base.junction_suspect and not crossing and
                        self.pending_sign is None and not sign_hint and
                        base.occlusion_fraction<.025)
                    if near_safe and self.side_probe_distance<1.2 and self.side_probe_motion_s<4.0:
                        cap=min(base.speed,3.0) if speed<=6 else 0.0
                        goal_angle=float(np.clip(base.angle,-3,3))
                        if obstacle['left']>w/2:goal_angle=min(goal_angle,0.0)
                        if obstacle['right']<w/2:goal_angle=max(goal_angle,0.0)
                        mode='vehicle_side_advance'
                        if speed>.5:
                            self.side_probe_distance+=speed/3.6*dt
                            self.side_probe_motion_s+=dt
            elif obstacle['bottom']>.62 and blocks_ego:
                cap=0
                mode='vehicle_in_turn_stop'
            if obstacle['bottom']>.84 and blocks_ego:
                cap=0
                mode='vehicle_too_close_stop'
        else:
            self.vehicle_growth.clear()
            self.gap_clear_s=0.0
            self.previous_gap_target=None
            self.avoid_clear_s+=dt
            if self.avoid_clear_s>.35:
                self.avoid_side=None
                self.side_probe_distance=0.0
                self.side_probe_motion_s=0.0
            if uncertain_side_contacts and not self.turn_direction:
                # Slow past an off-axis uncertain object using the road
                # controller's path. No shape/color/location blacklist.
                cap=min(cap,6.0)
                if speed>9.0:cap=0.0
                if mode=='road':mode='vehicle_contact_uncertain'
        obstacle_stop=mode in ('vehicle_in_turn_stop','vehicle_too_close_stop','vehicle_no_gap_stop')
        stages.append(dict(layer='obstacle',cap=float(cap),mode=mode))
        # Perception/edge stop always overrides a traffic maneuver. Do not
        # rotate blindly when a spray or an off-road camera hides the route.
        if base.mode in ('occlusion_hold','occlusion_stop') or base.confidence==0 or base.near_clearance<=0:
            goal_angle=base.angle
            cap=0
            mode='traffic_perception_stop'
        elif base.mode in ('occlusion_crawl','occlusion_release'):
            goal_angle=base.angle
            cap=min(cap,base.speed)
            mode=base.mode
        elif base.mode in ('occlusion_near_recovery','near_curve_tracking'):
            goal_angle=base.angle
            cap=min(cap,base.speed)
            mode=base.mode
            # A committed legal turn is not a new branch choice. Stopping
            # here would freeze yaw progress indefinitely despite a current
            # connected outgoing road and an already aligned wheel.
            harmless_remote=(obstacle is not None and obstacle['width']<.07
                and obstacle['bottom']<.60 and
                not (obstacle['left']-.04*w<=w/2<=obstacle['right']+.04*w))
            path=base.current_path
            committed_visible=(self.turn_direction!=0 and
                self.turn_wheel_ready and
                self.pending_sign not in ('turn_prohibition_unknown',) and
                base.mode=='near_curve_tracking' and path is not None
                and path.get('image_margin_px',0)>=.025*w
                and path.get('foot_road_fraction',0)>=.85
                and self.turn_direction*base.angle>0 and abs(base.yaw_rate)<35
                and base.occlusion_fraction<.12 and not sign_conflicts
                and not crossing and not mouth_open
                and (obstacle is None or harmless_remote)
                and not obstacle_stop and self.turn_progress<=115)
            if committed_visible:
                # Steering, margin and speed now use THIS observed branch.
                # A dimensionless ratio to a clipped viewport is not a new
                # hard boundary test: the path constructor already checks
                # current ground support and absolute image-edge margin.
                cap=min(base.speed,6.0) if speed<=9 else 0.0
                goal_angle=base.angle
                branch_owner=True
                if branch_reobserve:
                    self.turn_elapsed=0.0
                mode='turn_branch_tracking' if cap>0 else 'turn_branch_align'
            elif (not self.turn_direction and obstacle is None and not obstacle_stop
                    and not uncertain_side_contacts and not sign_conflicts
                    and not crossing and not mouth_open and not t_shape
                    and side_evidence is None and self.handover_remaining<=0
                    and self.sign_brake_limit is None and speed<=6
                    and self.pending_sign in ('turn_left','turn_right','no_left_turn',
                                              'no_right_turn','go_straight')):
                # A pending sign selects the NEXT branch. It must not freeze
                # an observed incoming-stem correction before any junction
                # opening is present. Keep the vote, not a second route.
                incoming_near_proof=current_near_motion_evidence(mask,raw_image,base,
                    wheel=actual_angle,speed=speed)
                if incoming_near_proof is not None:
                    cap=min(cap,base.speed,3.0)
                    mode='incoming_near_recovery'
                else:
                    cap=0.0;mode='occluded_traffic_stop'
            elif (self.turn_direction or obstacle is not None or
                    sign_conflicts or crossing or mouth_open or
                    self.pending_sign in ('turn_left','turn_right','no_left_turn',
                                          'no_right_turn','turn_prohibition_unknown')):
                cap=0.0
                mode='occluded_traffic_stop'
        elif base.mode in ('occlusion_track','occlusion_coast') and (self.turn_direction or
                self.pending_sign in ('turn_left','turn_right')):
            cap=0
            mode='occluded_junction_stop'
        # A mask_lost caused by missing straight/far scanlines is not an
        # empty Road mask. A committed turn can own CURRENT broad close ground
        # and steer within that same checked region, while real hazards veto.
        # Select ONE currently observed legal path, then evaluate steering,
        # wheel alignment, motion and white coverage on that same path.
        # A rejected physical motion check retains the path and brakes;
        # it cannot silently fall back to a different Road steering target.
        junction_path=current_junction_floor(mask,raw_image,self.turn_direction)
        if junction_path is None:
            junction_path=current_legal_branch_arc(mask,raw_image,self.turn_direction,
                actual_angle=actual_angle,emitted_angle=self.last_angle)
        contract=self.turn_contract.evaluate(junction_path,base,raw_image,
            shape=mask.shape,direction=self.turn_direction,
            committed_goal=committed_goal_angle,entry_braking=self.turn_entry_braking,
            progress=self.turn_progress,speed=speed,wheel=actual_angle,yaw=yaw_rate,
            dt=dt,elapsed=self.elapsed,
            legal=(self.pending_sign!='turn_prohibition_unknown' and not sign_conflicts),
            hazard=(obstacle is not None or obstacle_stop or uncertain_side_contacts>0))
        if contract['owns_path']:
            path=contract['path'];branch_owner=True
            goal_angle=contract['goal_angle'];cap=contract['speed_cap']
            mode=contract['mode']
            if branch_reobserve:
                self.turn_elapsed=0.0
            decision=replace(base,near_left=float(path['near_bounds'][0]),
                near_right=float(path['near_bounds'][1]),near_clearance=path['near_clearance'],
                confidence=2/3,target_x=float(path.get('target',base.target_x)),
                current_path=path)
        elif base.spray_prealert:
            cap=min(cap,base.speed)
            if cap==0:
                mode='spray_prealert_brake'
        # Rejecting path ownership must not expose a positive command left
        # over from the earlier turn schedule. These are route/feedback hard
        # vetoes, independent of which layer owns geometry.
        turn_hard_reason=None
        if self.turn_direction:
            if sign_conflicts:
                turn_hard_reason='sign_conflict_wait'
            elif self.pending_sign=='turn_prohibition_unknown':
                turn_hard_reason='restriction_direction_unknown_stop'
            elif actual_angle is None or not np.isfinite([speed,actual_angle,yaw_rate]).all():
                turn_hard_reason='turn_physical_feedback_invalid'
            if turn_hard_reason:
                cap=0.0;mode=turn_hard_reason
                contract['global_veto']=turn_hard_reason
                contract['cap_scope']='global_hard_stop'
        if obstacle is not None and 0<cap<=14 and speed>cap+3:
            cap=0.0
            mode='vehicle_approach_brake'
        # Turn/obstacle layers cannot restore ga above this active entry
        # braking decision. Their current steering remains authoritative.
        if self.sign_brake_limit is not None and cap>0:
            cap=0.0
            mode='sign_speed_brake'
        stages.append(dict(layer='perception',cap=float(cap),mode=mode))
        # When Traffic keeps Road's CURRENT close curve rather than choosing
        # a different route, use the producing path's validated slew. Road
        # checks physical wheel alignment against that same current path.
        # A second lower22deg/s cap otherwise prolongs its alignment brake.
        # Still bound the EMITTED step from Traffic's last actual request;
        # do not jump to Road's possibly advanced internal request history.
        owned=base.current_path or {}
        shared_curve_slew=bool(not self.turn_direction and not branch_owner
            and self.handover_remaining<=0 and obstacle is None
            and not uncertain_side_contacts and not sign_conflicts
            and not base.junction_suspect and not base.junction_hold_path
            and gap_target is None
            and abs(goal_angle-base.angle)<1e-9
            and owned.get('source')=='current_connected_near_curve'
            and len(owned.get('rows',[]))==4
            and owned.get('foot_road_fraction',0)>=.85
            and base.mode in ('near_curve_tracking','occlusion_near_recovery'))
        outgoing_contract=current_outgoing_contract(base,route_mask,raw_image,
            mode=mode,goal=goal_angle,previous_command=self.last_angle,
            wheel=actual_angle,speed=speed,yaw=yaw_rate,elapsed=self.elapsed,
            hazard=bool(self.turn_direction or branch_owner or obstacle is not None
                or uncertain_side_contacts or sign_conflicts or gap_target is not None))
        if outgoing_contract['applied']:
            # Steering and Road's measured-wheel cap refer to one proposal.
            # Road's own filter stays active; do not feed another layer's old
            # turn command back into its history for the next frame.
            angle=base.angle
            cap=min(cap,base.speed)
            decision=replace(decision,current_path=outgoing_contract['path'])
            rate=None
        elif mode=='road':
            angle=base.angle
            rate=None
        else:
            rate=(40.0 if base.mode=='near_curve_tracking' else 35.0) if shared_curve_slew else 30.0 if self.turn_direction else 22.0
            rate=self.turn_alignment_request.rate(rate,contract=contract,
                entry_braking=self.turn_entry_braking,direction=self.turn_direction,
                speed=speed,wheel=actual_angle,goal=goal_angle,dt=dt,cap=cap)
            angle=float(np.clip(goal_angle,self.last_angle-rate*dt,self.last_angle+rate*dt))
        if contract['owns_path']:
            # Speed must agree with both the selected goal and the command
            # actually emitted after slew. Do not accelerate while command
            # history still points down the previous corridor.
            emitted_error=abs(angle-contract['goal_angle'])
            contract['emitted_goal_error_deg']=float(emitted_error)
            if emitted_error>5 and cap>0:
                cap=0.0;mode='turn_junction_align'
                contract['cap_reason']='emitted_command_not_on_selected_path'
            contract['final_traffic_cap']=float(cap)
            contract['final_traffic_mode']=mode
        alignment_cap=self.turn_alignment_request.physical_cap(cap,
            contract=contract,direction=self.turn_direction,wheel=actual_angle,
            goal=goal_angle)
        if alignment_cap<cap:
            cap=alignment_cap;mode='turn_alignment_physical_brake'
        if contract['owns_path']:
            contract['final_traffic_cap']=float(cap)
            contract['final_traffic_mode']=mode
        if base.mode=='corner_momentum_brake' and not contract.get('owns_path'):
            cap=0.0
            mode='corner_momentum_brake'
        local_recovery_proofs=[c['off_current_corridor']
            for c in vehicle_contact_evidence if c.get('off_current_corridor')
            and c['off_current_corridor'].get('local_recovery_only')]
        local_recovery_owner=bool(local_recovery_proofs or incoming_near_proof)
        if local_recovery_owner:
            # Every retained hazard and physical veto above still applies.
            # Reuse Road's current target and physical cap for a short visible
            # recovery horizon; do not grant its ordinary avoidance speed.
            cap=min(cap,3.0)
            if mode=='road':mode='near_curve_remote_recovery'
            if abs(angle-base.angle)>5 and cap>0:
                cap=0.0;mode='incoming_near_align'
            stages.append(dict(layer='local_recovery_horizon',cap=float(cap),mode=mode))
        self.last_angle=angle
        self.info=dict(mode=mode,pending_sign=self.pending_sign,visible_sign=visible,
                       sign_conflicts=sign_conflicts,requested_turn=requested_turn,
                       sign_approach_reserve=dict(self.sign_approach_reserve.info),
                       sign_brake_revalidation=self.sign_brake_revalidation.info,
                       approach_limit=approach_limit,flow_approach_resolved=flow_approach_resolved,size_only_close_preview=size_only_close_preview,sign_size=self.sign_size,sign_brake_limit=self.sign_brake_limit,
                       crossing=crossing,junction_mouth_open=mouth_open,
                       junction_mouth_count=self.junction_mouth_count,
                       legal_side_count=self.legal_side_count,legal_side_evidence=side_evidence,
                       t_junction=t_shape,t_junction_count=self.t_junction_count,
                       turn_progress=self.turn_progress,
                       turn_direction=self.turn_direction,turn_entry_braking=self.turn_entry_braking,turn_wheel_ready=self.turn_wheel_ready,requested_goal_angle=float(goal_angle),obstacles_on_road=len(vehicles),
                       uncertain_side_contacts=uncertain_side_contacts,
                       vehicle_contacts=vehicle_contact_evidence,
                       local_recovery_owner=local_recovery_owner,
                       stopped_depth_recovery=dict(active=self.depth_recovery_active,
                           confirmed_stopped_s=self.depth_recovery_stopped_s,
                           activation_wait_s=.60,
                           affects_ordinary_moving_path=False),
                       local_recovery_horizon=dict(applied=local_recovery_owner,
                           request_ceiling=3.0 if local_recovery_owner else None,
                           current_proofs=local_recovery_proofs,
                           incoming_path_proof=incoming_near_proof),
                       side_advance_distance_m=self.side_probe_distance,
                       remote_vehicle_preview=remote_vehicle,obstacle_pass=pass_evidence,
                       shared_current_curve_slew=shared_curve_slew,emitted_slew_rate=(None if mode=='road' else rate),
                       avoid_side=self.avoid_side,gap_target=gap_target,
                       decision_stages=stages,traffic_cap=float(cap),
                       turn_alignment_request=self.turn_alignment_request.info,
                       branch_owner=branch_owner,branch_reobserve=branch_reobserve,
                       outgoing_path_contract=outgoing_contract,
                       turn_path_contract=contract,
                       selected_path=path if branch_owner else None,
                       stop_reason=mode if cap<=0 else None)
        if mode!='road':
            decision=replace(decision,speed=float(cap),angle=float(angle),mode=mode)
        return decision
