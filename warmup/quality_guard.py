"""Current-image road-edge evidence under bright, partial occlusion.

No map locations or stored road mask are used to fit the corridor. Missing
interior pixels are accepted only when the camera shows bright coverage.
"""
import cv2
import numpy as np


def current_white_mask(image):
    blue,green,red=cv2.split(image)
    low=cv2.min(cv2.min(blue,green),red)
    high=cv2.max(cv2.max(blue,green),red)
    return cv2.bitwise_and(cv2.inRange(low,211,255),cv2.inRange(cv2.subtract(high,low),0,34))!=0


def fill_enclosed_paint_holes(mask, bgr):
    """Repair tiny white road-marking holes, never connect outer islands.

    Candidate pixels must be fully enclosed by current Road, narrow and low
    in the camera, and visibly white. Large spray gaps/unknown objects remain
    background. This is control geometry; the saved model mask stays intact.
    """
    road=(mask==1).astype(np.uint8)
    if bgr is None:return road,0
    h,w=road.shape
    image=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    n,labels,stats,_=cv2.connectedComponentsWithStats(1-road)
    count=0
    for label in range(1,n):
        x,y,width,height,area=stats[label]
        if (x==0 or y<.50*h or x+width>=w or y+height>=h or
            width>9 or height>48 or area>256):continue
        hole=labels[y:y+height,x:x+width]==label
        if white[y:y+height,x:x+width][hole].mean()<.60:continue
        road[y:y+height,x:x+width][hole]=1
        count+=1
    return road,count


def current_near_curve_path(mask, bgr):
    """Visible close edges when a curved road leaves the forward scanlines.

    This camera ground strip is only an image safety proxy, not the vehicle
    footprint. Never extrapolate a hidden road across an empty camera bottom.
    Require current visible road next to the strip and one connected corridor.
    """
    if bgr is None:
        return None
    h,w=mask.shape
    image=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    foot=(slice(int(.988*h),h),slice(int(.49*w),int(.51*w)))
    if (mask[foot]==1).mean()<.85:
        return None
    # Use actual predicted components. A small isolated bottom patch must
    # not make the closest connected boundary appear farther away.
    road=(mask==1).astype(np.uint8)
    n,labels=cv2.connectedComponents(road)
    counts=np.bincount(labels[foot].ravel(),minlength=n);counts[0]=0
    if counts.max()<15:
        return None
    connected=labels==int(np.argmax(counts))
    near_region=connected[int(.94*h):]
    if near_region.sum()<.02*h*w or (white[int(.94*h):][near_region]).mean()>.65:
        return None
    bounds=[]
    for fraction in (.995,.97,.93,.89):
        y=min(h-1,int(fraction*h))
        row=connected[max(0,y-2):min(h,y+3)].sum(0)>(min(h,y+3)-max(0,y-2))//2
        changes=np.diff(np.r_[False,row,False].astype(np.int8))
        runs=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(changes==1),
              np.flatnonzero(changes==-1)) if b-a>.18*w]
        if len(runs)!=1:return None
        a,b=runs[0]
        if a<3 and b>w-4:return None
        bounds.append((a,b))
    near=bounds[0]
    margin=min(w/2-near[0],near[1]-w/2)
    if margin<.025*w:return None
    widths=np.array([b-a+1 for a,b in bounds],float)
    centers=np.array([(a+b)/2 for a,b in bounds],float)
    if np.any(np.diff(widths)>.08*w) or np.max(np.abs(np.diff(centers)))>.08*w:
        return None
    # A crossing or a second big branch is ambiguous even if the bottom
    # strip happens to show road. Do not select a branch by area.
    for fraction in (.70,.80):
        y=int(fraction*h)
        row=connected[y-2:y+3].sum(0)>2
        xs=np.flatnonzero(row)
        if len(xs)>8 and xs[-1]-xs[0]>.85*w:return None
        if np.count_nonzero(np.diff(np.r_[False,row,False].astype(np.int8))==1)>1:
            return None
    target=float(.3*centers[2]+.7*centers[3])
    clipped_right=all(b>w-4 for a,b in bounds)
    clipped_left=all(a<3 for a,b in bounds)
    minimum_angle=0.0
    if clipped_right and w/2-near[0]<.22*w:
        if target<=.58*w:return None
        minimum_angle=30.0
    if clipped_left and near[1]-w/2<.22*w:
        if target>=.42*w:return None
        minimum_angle=-30.0
    return dict(near_bounds=near,near_clearance=float(margin/widths[0]),
                center=float(centers[0]),target=target,width=float(widths[0]),
                target_width=float(widths[-1]),minimum_angle=minimum_angle,
                curve_recovery=True, rows=[dict(y=float(y),left=float(a),right=float(b)) for y,(a,b) in zip((.995,.97,.93,.89),bounds)], image_margin_px=float(margin), foot_road_fraction=float((mask[foot]==1).mean()), source="current_connected_near_curve")


def visible_spray_corridor(mask, bgr, *, require_spray=True):
    if bgr is None:
        return None
    h, w = mask.shape
    image = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    blue,green,red=cv2.split(image)
    low=cv2.min(cv2.min(blue,green),red)
    high=cv2.max(cv2.max(blue,green),red)
    white=(cv2.bitwise_and(cv2.inRange(low,211,255),
                          cv2.inRange(cv2.subtract(high,low),0,34))!=0)
    if white.mean() > .80 or (require_spray and white[int(.50*h):int(.93*h)].mean() < .07):
        return None
    road = mask == 1
    # The current camera foot region must still contain actual road pixels.
    foot = road[int(.88*h):int(.97*h),int(.43*w):int(.57*w)]
    if not foot.size or foot.mean() < .75:
        return None
    points=[]
    # Spray can hide the middle rows while the closer road edges remain
    # visible. Include those current observations in the same bounded fit;
    # all footprint, edge, junction and brightness checks still apply.
    for fraction in np.arange(.62,.881,.02):
        y=int(fraction*h)
        row=road[y-3:y+4].sum(0)>3
        changes=np.diff(np.r_[False,row,False].astype(np.int8))
        runs=[(int(a),int(b)-1) for a,b in zip(
            np.flatnonzero(changes==1),np.flatnonzero(changes==-1)) if b-a>=8]
        if not runs:
            continue
        a,b=runs[0][0],runs[-1][1]
        span=b-a+1
        if a<4 or b>w-5 or not .20*w<span<.86*w:
            continue
        if min(w/2-a,b-w/2)/span < .20 or row[a:b+1].mean()<.60:
            continue
        holes=[(left[1]+1,right[0]) for left,right in zip(runs,runs[1:])]
        if any(end-start>.40*span or white[y-3:y+4,start:end].mean()<.60
               for start,end in holes):
            continue
        points.append((y,a,b))
    if len(points)<4:
        return None
    data=np.asarray(points,dtype=float)
    ys=data[:,0]
    actual=road[int(.85*h)-3:int(.85*h)+4].sum(0)>3
    xs=np.flatnonzero(actual)
    if len(xs)<8:
        return None
    best=None
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            distance=ys[j]-ys[i]
            if distance<.05*h:
                continue
            slopes=(data[j,1:]-data[i,1:])/distance
            if not (-4<slopes[0]<1 and -1<slopes[1]<4
                    and .5<slopes[1]-slopes[0]<5):
                continue
            fitted=data[i,1:]+(ys-ys[i])[:,None]*slopes
            good=(np.abs(fitted-data[:,1:]).max(1)<.016*w)
            if good.sum()<4 or np.ptp(ys[good])<.06*h:
                continue
            near_estimate=np.clip(data[i,1:]+(.85*h-ys[i])*slopes,0,w-1)
            if (abs(xs[0]-near_estimate[0])>.04*w or
                    abs(xs[-1]-near_estimate[1])>.04*w):
                continue
            score=(int(good.sum()),float(np.ptp(ys[good])))
            if best is None or score>best[0]:
                best=(score,good)
    if best is None:
        return None
    good=best[1]
    coefficients=np.linalg.lstsq(np.c_[ys[good],np.ones(good.sum())],
                                   data[good,1:],rcond=None)[0]
    if abs(coefficients[0].mean())>1.5:
        return None
    def bounds(fraction):
        return np.array((fraction*h,1.0))@coefficients
    near=bounds(.85)
    clipped=np.clip(near,0,w-1)
    if (abs(xs[0]-clipped[0])>.04*w or abs(xs[-1]-clipped[1])>.04*w
            or min(w/2-clipped[0],clipped[1]-w/2)/(clipped[1]-clipped[0]+1)<.28):
        return None
    middle=bounds(.70)
    if abs(middle.mean()-w/2)>.18*w:
        return None
    # Reject a current wide junction even when a narrow incoming stem fits.
    xs70=np.flatnonzero(road[int(.70*h)-3:int(.70*h)+4].sum(0)>3)
    if len(xs70) and xs70[-1]-xs70[0]>.80*w:
        return None
    scan=np.arange(h)
    edges=np.c_[scan,np.ones(h)]@coefficients
    columns=np.arange(w)[None,:]
    corridor=(columns>=edges[:,0,None]+3)&(columns<edges[:,1,None]-2)
    corridor[:int(.55*h)]=False
    corridor[int(.93*h):]=False
    if corridor.sum()<100:
        return None
    fraction=float(white[corridor].mean())
    centers=tuple(float(bounds(y).mean()) for y in (.85,.70,.55))
    widths=tuple(float(np.diff(bounds(y))[0]) for y in (.85,.70,.55))
    return dict(current_fit_edge_rows=[dict(y=float(y/h),left=float(a),right=float(b)) for y,a,b in data[good]],near_bounds=tuple(float(x) for x in clipped), centers=centers,
                widths=widths, white_fraction=fraction,
                clearance=float(min(w/2-clipped[0],clipped[1]-w/2)/
                                (clipped[1]-clipped[0]+1)), points=int(good.sum()))


def current_near_water_path(mask, bgr, *, allow_white_foot=False):
    """Current close road evidence; no location, old mask or fountain class.

    Used for bounded slow recovery only. The closest camera footprint must
    be road and mostly free of white occlusion. A visible, connected near
    corridor can bend outside the central preview without implying that the
    vehicle footprint itself is already off road.
    """
    if bgr is None:
        return None
    h,w=mask.shape
    image=cv2.resize(bgr,(w,h),interpolation=cv2.INTER_LINEAR)
    white=current_white_mask(image)
    foot=(slice(int(.97*h),int(.995*h)),slice(int(.49*w),int(.51*w)))
    if (mask[foot]==1).mean()<.95 or (not allow_white_foot and white[foot].mean()>.25):
        return None
    road=cv2.morphologyEx((mask==1).astype('uint8'),cv2.MORPH_CLOSE,
                          np.ones((3,5),np.uint8))
    n,labels=cv2.connectedComponents(road)
    counts=np.bincount(labels[foot].ravel(),minlength=n);counts[0]=0
    if counts.max()<20:
        return None
    connected=labels==int(np.argmax(counts))
    chosen=[]
    for fraction in (.985,.93,.87,.83):
        y=int(fraction*h)
        row=connected[max(0,y-2):min(h,y+3)].sum(0)>2
        changes=np.diff(np.r_[False,row,False].astype(np.int8))
        runs=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(changes==1),
            np.flatnonzero(changes==-1)) if b-a>=8]
        runs=[ab for ab in runs if ab[1]-ab[0]+1>.18*w]
        if len(runs)!=1:
            return None
        a,b=runs[0]
        if a<3 and b>w-4:
            return None
        chosen.append((a,b))
    near=chosen[0];span=near[1]-near[0]+1
    margin=min(w/2-near[0],near[1]-w/2)
    if margin<max(.02*w,.035*span):
        return None
    # Current near edge observations must form one consistent path. Very
    # wide middle road could be a crossing; do not choose a branch here.
    middle=connected[int(.70*h)-2:int(.70*h)+3].sum(0)>2
    xs=np.flatnonzero(middle)
    if len(xs)>8 and xs[-1]-xs[0]>.85*w:
        return None
    widths=np.array([b-a+1 for a,b in chosen],dtype=float)
    centers=np.array([(a+b)/2 for a,b in chosen])
    if (widths[-1]>1.15*widths[0] or
        np.max(np.abs(np.diff(centers)))>.15*w or
        np.max(np.abs(np.diff(widths)))>.25*w):
        return None
    target=float(.35*centers[1]+.65*centers[2])
    if ((w/2-near[0]<.08*w and target<w/2) or
            (near[1]-w/2<.08*w and target>w/2)):
        return None

    return dict(near_bounds=near,near_clearance=float(margin/span),
                center=float(centers[0]),target=target,
                width=float(widths[0]),target_width=float(widths[2]))
