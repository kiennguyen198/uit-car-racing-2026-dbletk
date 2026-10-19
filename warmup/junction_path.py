"""Current broad junction ground supports a voted turn, never a hidden path."""
import cv2
import numpy as np
from quality_guard import current_white_mask, fill_enclosed_paint_holes

def current_junction_floor(mask, raw, direction):
    if raw is None or direction not in (-1,1):return None
    h,w=mask.shape
    road=mask==1
    # Do not accept a full-image hallucination: a junction has an observed
    # road/sky boundary. No old mask or stored scene state supplies free space.
    if road[:int(.45*h)].mean()>.15:return None
    rows=[]
    for fraction in (.85,.89,.93,.97,.995):
        y=min(h-1,int(fraction*h));band=road[max(0,y-2):min(h,y+3)]
        center=band[:,int(.34*w):int(.66*w)]
        if center.size==0 or center.mean()<.95:return None
        xs=np.flatnonzero(band.mean(0)>.80)
        if len(xs)<.78*w or xs[0]>.10*w or xs[-1]<.90*w:return None
        rows.append(dict(y=fraction,left=int(xs[0]),right=int(xs[-1])))
    image=cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    observed=(slice(int(.85*h),h),slice(int(.10*w),int(.90*w)))
    if white[observed].mean()>.10:return None
    near=rows[-1];margin=min(w/2-near['left'],near['right']-w/2)
    return dict(source='current_broad_junction_floor',direction=direction,
                near_bounds=[near['left'],near['right']],image_margin_px=float(margin),
                near_clearance=float(margin/max(1,near['right']-near['left']+1)),
                foot_road_fraction=float(road[observed].mean()),
                current_white_fraction=float(white[observed].mean()),rows=rows,
                meaning='Current close free region for an already voted legal turn, not an unseen outgoing lane')


def current_legal_branch_arc(mask, raw, direction, *, actual_angle=None, emitted_angle=None):
    """Short curved image corridors entirely on CURRENT observed Road.

    A legal turn need not keep the forward optical scanline inside its road.
    Sample a small lattice of curves anchored at current camera ground, then
    require clearance along every point. Image radii are safety proxies,
    not calibrated physical footprints or hidden-road extrapolation.
    """
    if raw is None or direction not in (-1,1):return None
    h,w=mask.shape
    original=mask==1
    if original[:int(.45*h)].mean()>.15:return None
    image=cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    foot=(slice(int(.965*h),h),slice(int(.47*w),int(.53*w)))
    if original[foot].mean()<.95 or white[foot].mean()>.12:return None
    repaired,_=fill_enclosed_paint_holes(mask,raw)
    n,labels=cv2.connectedComponents(repaired)
    counts=np.bincount(labels[foot].ravel(),minlength=n);counts[0]=0
    if counts.max()<.02*h*w*.01:return None
    road=(labels==int(np.argmax(counts))).astype('uint8')
    clearance=cv2.distanceTransform(road,cv2.DIST_L2,3)
    candidates=[]
    samples=np.linspace(0,1,25)
    for goal_y in (.60,.67,.74,.80):
        ys=.99-samples*(.99-goal_y)
        for offset in (.12,.20,.28,.36,.44):
            for tangent in (0.,.04,.08):
                xs=w/2+direction*w*(tangent*samples+(offset-tangent)*samples*samples)
                radii=w*(.025*(1-samples)+.012*samples)
                ix=np.rint(xs).astype(int);iy=np.rint(ys*h).astype(int)
                if np.any(ix-radii<2) or np.any(ix+radii>w-3):continue
                distance=clearance[iy,ix]
                if np.any(distance<radii):continue
                covered=False
                for x,y,radius in zip(ix,iy,radii):
                    patch=white[max(0,y-2):min(h,y+3),int(x-radius):int(x+radius)+1]
                    if not patch.size or patch.mean()>.12:covered=True;break
                if covered:continue
                minimum=float(np.min(distance-radii))
                score=4*(.99-goal_y)*w+minimum+.25*float(distance.mean())-.10*offset*w
                candidates.append((score,goal_y,offset,tangent,xs,ys,radii,distance))
    if not candidates:return None
    selected=max(candidates,key=lambda c:c[0])
    geometric_best=selected
    feasible=[]
    if (actual_angle is not None and emitted_angle is not None and
            np.isfinite([actual_angle,emitted_angle]).all()):
        # All alternatives already passed the SAME current connected-road,
        # full tube clearance and white tests. Pick the longest/highest-score
        # currently observed arc the physical wheel and emitted request can
        # both follow. No interpolation into unobserved ground or old mask.
        for c in candidates:
            goal=float(np.clip(direction*70*(.39/max(.19,.99-c[1]))*c[2],-25,25))
            if abs(actual_angle-goal)<=5 and abs(emitted_angle-goal)<=5:
                feasible.append(c)
        if feasible:selected=max(feasible,key=lambda c:c[0])
    _,goal_y,offset,tangent,xs,ys,radii,distance=selected
    row=road[int(.99*h)]!=0;where=np.flatnonzero(row)
    if not len(where):return None
    near=[int(where[0]),int(where[-1])]
    margin=min(w/2-near[0],near[1]-w/2)
    return dict(source='current_legal_branch_arc_lattice',direction=direction,
        near_bounds=near,image_margin_px=float(margin),
        near_clearance=float(margin/max(1,near[1]-near[0]+1)),
        foot_road_fraction=float(original[foot].mean()),
        current_white_fraction=float(white[foot].mean()),
        target=float(xs[-1]),target_y=goal_y,
        curve_offset_fraction=offset,curve_initial_tangent_fraction=tangent,
        minimum_tube_clearance_px=float(np.min(distance-radii)),
        rows=[dict(y=float(y),center=float(x),half_width_px=float(r),
                   current_clearance_px=float(d)) for x,y,r,d in zip(xs,ys,radii,distance)],
        selection_policy='current_physical_and_emitted_feasible' if feasible else 'current_geometric_best_requires_alignment',
        physical_feasible_candidates=len(feasible),current_candidates=len(candidates),
        default_target_y=float(geometric_best[1]),default_offset_fraction=float(geometric_best[2]),
        selected_target_y=float(goal_y),selected_offset_fraction=float(offset),
        meaning='Current observed curved free corridor for the voted legal branch; no coordinates/checkpoint/route history')
