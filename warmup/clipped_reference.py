"""Steering-only estimate from CURRENT uncut road widths.

The hidden edge is explicitly estimated. Observed bounds are never widened,
and this helper grants no speed or motion permission.
"""
import numpy as np


def current_clipped_reference(road,near_bounds,*,near_y=.85):
    h,w=road.shape
    left,right=near_bounds
    cut_left=left<=1
    cut_right=right>=w-2
    if cut_left==cut_right:
        return None
    foot=road[int(.965*h):int(.995*h),int(.47*w):int(.53*w)]
    if not foot.size or foot.mean()<.95:
        return None
    observed=[]
    for fraction in np.arange(.56,.841,.02):
        y=int(fraction*h)
        row=road[max(0,y-2):min(h,y+3)].mean(0)>.6
        transitions=np.diff(np.r_[False,row,False].astype(np.int8))
        runs=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(transitions==1),np.flatnonzero(transitions==-1)) if b-a>.12*w]
        if len(runs)!=1:
            continue
        a,b=runs[0]
        if a<4 or b>w-5 or not .15*w<b-a+1<.90*w:
            continue
        if row[a:b+1].mean()<.95:
            continue
        observed.append((fraction,a,b))
    if len(observed)<5:
        return None
    data=np.asarray(observed,float)
    ys=data[:,0]
    widths=data[:,2]-data[:,1]+1
    best=None
    for i in range(len(ys)):
        for j in range(i+1,len(ys)):
            span=ys[j]-ys[i]
            if span<.10:
                continue
            slope=(widths[j]-widths[i])/span
            if not .5*w<slope<5*w:
                continue
            prediction=widths[i]+slope*(ys-ys[i])
            good=np.abs(widths-prediction)<.025*w
            if good.sum()<5 or np.ptp(ys[good])<.12:
                continue
            score=(int(good.sum()),float(np.ptp(ys[good])))
            if best is None or score>best[0]:best=(score,good)
    if best is None:
        return None
    good=best[1]
    coeff=np.polyfit(ys[good],widths[good],1)
    expected=float(np.polyval(coeff,near_y))
    observed_width=right-left+1
    horizon=float(ys[good].max())
    if (near_y-horizon>.12 or expected<observed_width+.02*w
        or expected>1.5*w or expected-observed_width>.30*w):
        return None
    # The currently visible edge must agree with the same width-support
    # region. Do not borrow a width from a different overlapping branch.
    column=2 if cut_left else 1
    edge=np.polyfit(ys[good],data[good,column],2)
    edge_error=float(np.percentile(np.abs(np.polyval(edge,ys[good])-data[good,column]),95))
    visible=right if cut_left else left
    if edge_error>.025*w or abs(np.polyval(edge,near_y)-visible)>.04*w:
        return None
    center=float(right-(expected-1)/2 if cut_left else left+(expected-1)/2)
    return dict(source='current_two_edge_width_clipped_reference',center=center,
        observed_bounds=[float(left),float(right)],observed_width=float(observed_width),
        estimated_width=expected,missing_edge_estimated=True,observed_y_span=[float(ys[good].min()),horizon],
        fit_rows=[dict(y=float(y),left=float(a),right=float(b)) for y,a,b in data[good]],
        width_coefficients=list(map(float,coeff)),visible_edge_fit_error95_px=edge_error,
        speed_permission=False,observed_safety_bounds_unchanged=True)
