"""Obstacle relevance scoped to Road's currently observed near-motion tube.

Not an object whitelist and not permission to accelerate through an unseen
far road. New relevance cases retain a3 request and the producer's existing
motion budget, checked again on each current observation.
"""
import cv2
import numpy as np
from quality_guard import current_white_mask


def current_near_motion_evidence(mask, raw, base, *, wheel, speed):
    path=base.current_path
    if (raw is None or base.mode!='near_curve_tracking' or path is None
        or path.get('source')!='current_connected_near_curve'
        or not 0 < base.speed <= 10 or base.junction_suspect
        or base.occlusion_fraction>=.12 or abs(base.yaw_rate)>=35
        or wheel is None or not np.isfinite([wheel,speed,base.angle]).all()
        or speed>9 or abs(float(wheel)-base.angle)>5):
        return None
    h,w=mask.shape;rh,rw=raw.shape[:2]
    if (mask==1)[:int(.45*h)].mean()>.15:
        return None
    rows=path.get('rows',[])
    if (len(rows)!=4 or path.get('foot_road_fraction',0)<.95
        or path.get('image_margin_px',0)<.04*w):
        return None
    horizon=min(r['y'] for r in rows)
    target=float(path['target'])
    if not .80<=horizon<=.90 or not np.isfinite(target):
        return None
    image=cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    proof=[]
    for fraction in np.linspace(horizon,.995,19):
        mix=(.995-fraction)/(.995-horizon)
        center=(1-mix)*w/2+mix*target;radius=.04*w
        a=int(center-radius);b=int(center+radius)+1
        if a<0 or b>w:
            return None
        y=min(h-1,int(fraction*h))
        ground=mask[max(0,y-2):min(h,y+3),a:b]==1
        cover=white[max(0,y-2):min(h,y+3),a:b]
        if not ground.size or ground.mean()<.90 or cover.mean()>.12:
            return None
        proof.append(dict(y=float(fraction),center=float(center),half_width_px=radius,
            road_fraction=float(ground.mean()),white_fraction=float(cover.mean())))
    return dict(source='same_current_near_motion_horizon',horizon_y=float(horizon),
        existing_target=target,rows=proof,scope='existing_bounded_near_motion_only',
        detector_class_unchanged=True,road_speed_cap_unchanged=True,
        existing_motion_budget_unchanged=True,
        revalidated_from_current_pixels=True)


def remote_box_outside_current_near_motion(mask, raw, box, base, *, wheel, speed,
                                         allow_depth_recovery=False):
    proof=current_near_motion_evidence(mask,raw,base,wheel=wheel,speed=speed)
    if proof is None:return None
    h,w=mask.shape;rh,rw=raw.shape[:2]
    x1,y1,x2,y2=map(float,box)
    if (not np.isfinite([x1,y1,x2,y2]).all() or not 0<=x1<x2<=rw
        or not 0<=y1<y2<=rh):return None
    horizon=proof['horizon_y']
    left=x1*w/rw-.04*w;right=x2*w/rw+.04*w
    if y2/rh>=.58:
        # A side object can be nearer than the distant-object threshold yet
        # still outside the ENTIRE observed correction tube. Certify lateral
        # separation at every strip, including the camera-foot anchor. This
        # also handles a horizontally clipped side box: the visible inward
        # face must remain beyond the padded tube; unseen space is not used.
        # This does not choose the avoidance gap or promise free far road.
        if speed>6 or y2/rh>=.985:return None
        separation=min(max(left-(row['center']+row['half_width_px']),
                           (row['center']-row['half_width_px'])-right)
                       for row in proof['rows'])
        if separation<=0:
            if not allow_depth_recovery:return None
            # A confirmed stopped episode may use the SAME current near
            # tube when an object's padded rectangle is beyond its depth.
            # Compare the actual five-pixel strips used by the pixel proof;
            # do not flatten every object onto the closest steering row.
            top=y1/rh-.025;bottom=y2/rh+.08
            overlapping=[]
            for row in proof['rows']:
                y=min(h-1,int(row['y']*h))
                strip_top=max(0,y-2)/h;strip_bottom=min(h,y+3)/h
                if top<=strip_bottom and bottom>=strip_top:
                    margin=max(left-(row['center']+row['half_width_px']),
                               row['center']-row['half_width_px']-right)
                    if margin<=0:return None
                    overlapping.append(float(margin))
            return dict(proof,box_bottom=y2/rh,box_width_fraction=(x2-x1)/rw,
                box_disjoint=True,rectangle_disjoint=True,
                depth_recovery_only=True,box_expansion_px=.04*w,
                box_bottom_expansion_fraction=.08,
                box_rows_with_depth_overlap=len(overlapping),
                minimum_overlapping_box_separation_px=min(overlapping) if overlapping else None,
                relevance_reason='expanded_rectangle_disjoint_from_current_near_tube',
                local_recovery_only=True,request_ceiling=3.0)
        return dict(proof,box_bottom=y2/rh,box_width_fraction=(x2-x1)/rw,
            box_disjoint=True,minimum_lateral_separation_px=float(separation),
            box_expansion_px=.04*w,
            relevance_reason='expanded_box_laterally_disjoint_from_current_near_tube',
            horizontally_clipped=bool(x1==0 or x2==rw),
            local_recovery_only=True,request_ceiling=3.0)
    if y2/rh+.025>=horizon:return None
    legacy=(base.speed<=6 and (x2-x1)/rw<.055 and not left<=w/2<=right)
    local_recovery_only=not legacy
    # Size/axis alone cannot veto a much closer observed recovery tube.
    # Retain the warning for the far route: the expanded certificate only
    # permits a lower3 request on the existing2m/3s Road recovery budget.
    if local_recovery_only and (speed>6 or y2/rh+.08>=horizon):return None
    return dict(proof,box_bottom=y2/rh,box_width_fraction=(x2-x1)/rw,
        local_recovery_only=bool(local_recovery_only),
        request_ceiling=3.0 if local_recovery_only else None)
