"""Fast diagnostic motion floor ONLY on a current observed corridor.

Never a minimum actual speed through hidden road. No map coordinates,
checkpoint, stored route, simulator state or rear-camera assumption.
"""
from dataclasses import replace
import cv2
import numpy as np
from quality_guard import current_near_water_path, current_near_curve_path, current_white_mask


class VisibleMotionPolicy:
    def __init__(self):
        self.motion_s=0.0
        self.distance_m=0.0
        self.clear_s=0.0
        self.target=None
        self.agreement_s=0.0
        self.info={}
        self.elapsed=0.0
        self.path_time=-1e9

    def apply(self, decision, base, traffic, mask, raw, state, *, dt_s, fresh):
        dt=float(np.clip(dt_s,.01,.25));speed=float(state.get('Speed',0));self.elapsed+=dt
        self.info={'reason':'existing_positive_or_hard_stop','motion_s':self.motion_s,'distance_m':self.distance_m}
        if decision.speed>=3:
            clear=base.confidence==1 and base.preview_quality>=.9 and base.occlusion_fraction<.12 and base.near_clearance>.30
            self.clear_s=self.clear_s+dt if clear else 0
            if self.clear_s>.5:
                self.motion_s=self.distance_m=0.0
            return decision
        hard=(not fresh or decision.mode in ('status_stale','final_status_stale','manual_assist',
              'vehicle_too_close_stop','vehicle_in_turn_stop','vehicle_no_gap_stop',
              'sign_conflict_wait','restriction_direction_unknown_stop','turn_reobserve',
              'fountain_physical_brake','corner_momentum_brake',
              'turn_align','turn_edge_brake','sign_brake','vehicle_approach_brake')
              or traffic.get('sign_conflicts') or traffic.get('obstacles_on_road',0)
              or traffic.get('turn_direction') or traffic.get('pending_sign')
              or traffic.get('crossing') or traffic.get('junction_mouth_open')
              or base.edge_trend_guard or (base.boundary_recovery_guard and base.near_clearance<.25))
        if hard:
            self.info['reason']='actual_traffic_or_stale_hard_veto';return decision
        path=current_near_water_path(mask,raw)
        if path is None:path=current_near_curve_path(mask,raw)
        if path is None:
            self.info['reason']='no_current_observed_corridor';return decision
        h,w=mask.shape;im=cv2.resize(raw,(w,h));white=current_white_mask(im)
        # Mask on white spray can be a hallucinated road. Require both
        # current near edge interiors to remain visibly unoccluded.
        paired=0
        for fraction in (.985,.93,.87,.83):
            y=int(fraction*h);xs=np.flatnonzero(mask[y]==1)
            if len(xs)<8:continue
            visible=[]
            for x in (int(xs[0])+4,int(xs[-1])-4):
                strip=white[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]
                visible.append(strip.size and strip.mean()<.40)
            paired+=all(visible)
        if paired<2 or path['near_clearance']<.25 or abs(base.yaw_rate)>25:
            self.info['reason']='near_edges_hidden_or_boundary_risk';return decision
        target=float(path['target'])
        self.agreement_s=self.agreement_s+dt if self.target is not None and self.elapsed-self.path_time<=.20 and abs(target-self.target)<.06*w else 0.0
        self.path_time=self.elapsed
        self.target=target
        if self.agreement_s<.08:
            self.info['reason']='confirm_current_path';return decision
        angle=float(np.clip(20*(target-w/2)/(w/2)-.08*base.yaw_rate,-12,12))
        if abs(float(state.get('Angle',0))-angle)>7 or speed>6:
            self.info['reason']='actual_wheel_or_speed_not_ready';return decision
        if self.motion_s>=2 or self.distance_m>=1.5:
            # Re-evaluate against the CURRENT observation already checked
            # above. Never renew an old mask/angle when the road is hidden.
            if self.agreement_s<.25 or path['near_clearance']<.30:
                self.info['reason']='visible_motion_budget_reobserve';return decision
            self.motion_s=self.distance_m=0.0
            self.info['budget_revalidated_current_edges']=True
        self.motion_s+=dt if speed>.5 else 0.0
        self.distance_m+=speed/3.6*dt
        request=6.0 if speed<1 and self.motion_s<.4 else 3.0
        self.info={'reason':'current_visible_path_motion_floor','request':request,'motion_s':self.motion_s,'distance_m':self.distance_m,'paired_current_visible_edges':paired,'target':target}
        return replace(decision,speed=request,angle=angle,mode='visible_path_advance')
