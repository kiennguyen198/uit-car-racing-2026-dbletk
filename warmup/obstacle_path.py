"""Current image corridor for an existing selected avoidance gap."""
import cv2
import numpy as np
from quality_guard import current_white_mask

def observed_gap_corridor(route_mask, raw, gap, bottom):
    if raw is None or not .43 < bottom < .80:return None
    h,w=route_mask.shape;lo,hi=gap
    if hi-lo < .07*w:return None
    target=(lo+hi)/2
    white=current_white_mask(cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR))
    goal_y=float(np.clip(bottom+.025,.48,.82))
    rows=[]
    for fraction in np.linspace(goal_y,.985,7):
        mix=(.985-fraction)/max(.01,.985-goal_y)
        center=(1-mix)*w/2+mix*target
        radius=(1-mix)*.025*w+mix*min(.018*w,(hi-lo)*.22)
        y=min(h-1,int(fraction*h));a=max(0,int(center-radius));b=min(w,int(center+radius)+1)
        strip=route_mask[max(0,y-2):min(h,y+3),a:b]
        cover=white[max(0,y-2):min(h,y+3),a:b]
        if not strip.size or strip.mean()<.90 or cover.mean()>.10:return None
        rows.append(dict(y=float(fraction),center=float(center),half_width=float(radius),road_fraction=float(strip.mean())))
    return dict(source='current_selected_avoidance_gap',target=float(target),gap=[float(lo),float(hi)],rows=rows)


def close_box_off_current_corridor(mask, raw, box, base, *, actual_angle=None, allow_early=False):
    """Require a currently observed road tube disjoint from a close box.

    No shape/color whitelist or reinterpretation of the detector class.
    Every sampled tube strip must be current, unoccluded Road. This only
    removes an irrelevant Traffic obstacle; it cannot raise the Road cap.
    Unobserved/ambiguous paths and a box touching the tube remain hazards.
    """
    if raw is None or base.mode in ('manual_assist','status_stale','final_status_stale',
            'mask_lost','occlusion_hold','occlusion_stop'):
        return None
    h,w=mask.shape;rh,rw=raw.shape[:2]
    x1,y1,x2,y2=map(float,box)
    if base.confidence!=1 or base.junction_suspect:
        return None
    if y2/rh<=.80:
        # Early relevance only on a resolved observed ego corridor while
        # the PHYSICAL wheel follows it. This retains Road's speed cap.
        if (not allow_early or y2/rh<=.55 or base.mode!='normal'
            or base.speed<=0 or base.preview_quality<.85
            or abs(base.angle)>=6 or abs(base.yaw_rate)>=8
            or actual_angle is None or abs(float(actual_angle)-base.angle)>4
            or base.geometry_guard or base.edge_trend_guard
            or base.boundary_recovery_guard or base.steering_settle_guard
            or base.mask_shift_guard):
            return None
    if not all(c is not None for c in base.centers):return None
    # These are the SAME current scanline targets used by Road. The bottom
    # anchor is the current camera axis, not a remembered lane center.
    ys=np.array([.55,.70,.85,.995])
    xs=np.array([base.centers[2],base.centers[1],base.centers[0],w/2],float)
    if not np.isfinite(xs).all():return None
    image=cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    if white[int(.55*h):].mean()>.55:return None
    left=x1*w/rw-.025*w;right=x2*w/rw+.025*w
    top=max(.55,y1/rh-.005);bottom=min(.995,y2/rh+.005)
    fractions=np.unique(np.r_[np.linspace(.55,.995,19),top,bottom])
    rows=[];minimum_separation=float('inf')
    for fraction in fractions:
        center=float(np.interp(fraction,ys,xs));radius=.04*w
        a=int(center-radius);b=int(center+radius)+1
        if a<0 or b>w:return None
        y=min(h-1,int(fraction*h))
        strip=mask[max(0,y-2):min(h,y+3),a:b]==1
        cover=white[max(0,y-2):min(h,y+3),a:b]
        if not strip.size or strip.mean()<.90 or cover.mean()>.12:return None
        in_box=top<=fraction<=bottom
        if in_box:
            separation=max(left-(center+radius),(center-radius)-right)
            if separation<=0:return None
            minimum_separation=min(minimum_separation,separation)
        rows.append(dict(y=float(fraction),center=center,half_width_px=radius,
            road_fraction=float(strip.mean()),white_fraction=float(cover.mean())))
    if not np.isfinite(minimum_separation):return None
    return dict(source='current_road_scanline_corridor',box_disjoint=True,
        minimum_box_separation_px=minimum_separation,box_expansion_px=.025*w,
        early_relevance=bool(y2/rh<=.80),
        rows=rows,detector_class_unchanged=True)


def close_box_off_current_near_curve(mask,raw,box,base):
    """Close-box relevance on Road's existing current curve, not a new path.

    Only the observed near horizon is certified. Retain the existing <=6
    request: this cannot license acceleration or unseen farther traversal.
    A true box touching the padded tube remains a hazard even on Road pixels.
    """
    path=base.current_path
    if (raw is None or base.mode!='near_curve_tracking' or path is None or
        path.get('source')!='current_connected_near_curve' or
        base.junction_suspect or abs(base.yaw_rate)>=35 or
        base.occlusion_fraction>=.12 or base.speed>6):return None
    h,w=mask.shape;rh,rw=raw.shape[:2]
    if (mask==1)[:int(.45*h)].mean()>.15:return None
    if path.get('foot_road_fraction',0)<.95 or path.get('image_margin_px',0)<.04*w:return None
    x1,y1,x2,y2=map(float,box)
    if not .80<y2/rh<.985:return None
    goal_y=min(r['y'] for r in path['rows'])
    if goal_y>.90 or goal_y<.80:return None
    target=float(path['target'])
    if not np.isfinite(target):return None
    white=current_white_mask(cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR))
    left=x1*w/rw-.025*w;right=x2*w/rw+.025*w
    top=max(goal_y,y1/rh-.005);bottom=min(.995,y2/rh+.005)
    fractions=np.unique(np.r_[np.linspace(goal_y,.995,19),top,bottom])
    minimum=float('inf');rows=[]
    for fraction in fractions:
        mix=(.995-fraction)/(.995-goal_y)
        center=(1-mix)*w/2+mix*target;radius=.04*w
        a=int(center-radius);b=int(center+radius)+1
        if a<0 or b>w:return None
        y=min(h-1,int(fraction*h))
        road=mask[max(0,y-2):min(h,y+3),a:b]==1
        cover=white[max(0,y-2):min(h,y+3),a:b]
        if not road.size or road.mean()<.90 or cover.mean()>.12:return None
        if top<=fraction<=bottom:
            separation=max(left-(center+radius),(center-radius)-right)
            if separation<=0:return None
            minimum=min(minimum,separation)
        rows.append(dict(y=float(fraction),center=float(center),half_width_px=radius,
            road_fraction=float(road.mean()),white_fraction=float(cover.mean())))
    if not np.isfinite(minimum):return None
    return dict(source='same_current_near_curve_corridor',box_disjoint=True,
        minimum_box_separation_px=float(minimum),box_expansion_px=.025*w,
        horizon_y=goal_y,existing_target=target,rows=rows,
        detector_class_unchanged=True,road_speed_cap_unchanged=True)
