"""Short image-geometry memory revalidated by CURRENT ground and one edge.

The missing edge is explicitly estimated, never described as observed. No
stored world position, checkpoint or scene schedule is used. Fully opaque
ground, an expired snapshot, turning or absent current support cannot move.
"""
import math
import cv2
import numpy as np
from quality_guard import current_white_mask


class FountainMemory:
    def __init__(self):
        self.travel_m=0.0
        self.snapshot=None
        self.info={}

    def clear(self):
        self.snapshot=None

    def advance(self,status,dt,fresh):
        if not fresh:
            self.clear()
            return
        self.travel_m+=max(0.,float(status.get('Speed',0)))/3.6*dt

    def arm(self,path,status,now):
        # Large heading corrections/turns do not have a straight corridor
        # memory. The snapshot is based on current two-edge geometry only.
        if path.get('paired_current_visible_edges',0)<2:return
        centers=path.get('centers');widths=path.get('widths')
        if not centers or not widths or abs(centers[1]-192)>.15*384:return
        self.snapshot=dict(time=now,travel_m=self.travel_m,
            heading=float(status.get('Heading',0)),path=dict(path))

    def current_support(self,mask,raw,status,now):
        self.info=dict(reason='no_recent_two_edge_snapshot')
        old=self.snapshot
        if old is None:return None
        age=now-old['time'];distance=self.travel_m-old['travel_m']
        delta=(float(status.get('Heading',0))-old['heading']+180)%360-180
        self.info.update(age_s=age,distance_m=distance,heading_change_deg=delta)
        if age>.50 or distance>1.50 or abs(delta)>3:
            self.info['reason']='snapshot_motion_budget_expired';return None
        h,w=mask.shape;image=cv2.resize(raw,(w,h));white=current_white_mask(image)
        foot=(slice(int(.965*h),int(.995*h)),slice(int(.44*w),int(.56*w)))
        if (mask[foot]==1).mean()<.85 or white[foot].mean()>.40 or white[int(.55*h):].mean()>.80:
            self.info['reason']='current_close_ground_hidden_or_missing';return None
        p=old['path'];centers=np.array(p['centers'],float);widths=np.array(p['widths'],float)
        focal=w/(2*math.tan(math.radians(60)/2))
        yaw_shift=-focal*math.radians(delta)
        road=cv2.morphologyEx((mask==1).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,5),np.uint8))
        observed={'left':[],'right':[]}
        for fraction in (.90,.86,.82,.78):
            y=min(h-1,int(fraction*h));mix=(fraction-.70)/.15
            center=centers[1]+mix*(centers[0]-centers[1])+yaw_shift
            width=widths[1]+mix*(widths[0]-widths[1])
            expected=(center-width/2,center+width/2)
            row=road[max(0,y-2):min(h,y+3)].mean(0)>.80
            changes=np.diff(np.r_[False,row,False].astype(np.int8))
            runs=[(int(a),int(b)-1) for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)) if b-a>.12*w]
            for side,index in (('left',0),('right',1)):
                candidates=[r[index] for r in runs if 4<r[index]<w-5 and abs(r[index]-expected[index])<.04*w]
                for x in candidates:
                    a,b=(x+5,x+17) if side=='left' else (x-16,x-4)
                    support=mask[max(0,y-2):min(h,y+3),a:b]==1
                    cover=white[max(0,y-2):min(h,y+3),a:b]
                    if support.size and support.mean()>.75 and cover.mean()<.35:
                        observed[side].append(dict(y=fraction,x=x,residual=float(x-expected[index])))
                        break
        side=max(observed,key=lambda s:len(observed[s]));points=observed[side]
        if len(points)<3:
            self.info['reason']='insufficient_current_visible_edge';return None
        residual=np.array([x['residual'] for x in points]);correction=float(np.median(residual))
        if np.max(np.abs(residual-correction))>.015*w:
            self.info['reason']='current_edge_disagrees_with_recent_shape';return None
        current_centers=centers+yaw_shift+correction
        near=(current_centers[0]-widths[0]/2,current_centers[0]+widths[0]/2)
        margin=min(w/2-near[0],near[1]-w/2)
        clearance=margin/max(1,widths[0]);target=float(current_centers[1])
        if clearance<.35 or abs(target-w/2)>.15*w:
            self.info['reason']='predicted_margin_or_heading_insufficient';return None
        self.info.update(reason='current_edge_revalidates_recent_shape',side=side,
            current_edge_points=points,missing_edge_estimated=True,
            close_ground_fraction=float((mask[foot]==1).mean()))
        return dict(source='recent_shape_current_edge_revalidated',flowing=True,
            centers=tuple(float(x) for x in current_centers),widths=tuple(float(x) for x in widths),
            near_bounds=tuple(float(x) for x in near),near_clearance=float(clearance),
            center=float(current_centers[0]),target=target,width=float(widths[0]),target_width=float(widths[1]),
            paired_current_visible_edges=0,missing_edge_estimated=True,
            memory_age_s=age,memory_distance_m=distance,current_visible_edge_side=side)
