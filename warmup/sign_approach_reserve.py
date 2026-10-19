"""Optical approach reserve for a voted mandatory turn, never route selection.

Online face growth and integrated actual speed estimate a local optical scale.
No physical sign dimensions, map coordinates, checkpoints or scene clocks.
The proposal can only activate the existing lower approach ceiling earlier.
"""
from collections import deque
import math
import statistics


class SignApproachReserve:
    def __init__(self):
        self.elapsed=0.0
        self.travel=0.0
        self.previous_speed=None
        self.shape=None
        self.kind=None
        self.samples=deque()
        self.info={}

    def clear(self):
        self.samples.clear()
        self.kind=None
        self.previous_speed=None

    def update(self, detections, *, shape, route, turning, speed, wheel, yaw, dt, fresh):
        self.info=dict(applied=False,reason='no_current_turn_approach_reserve',limit=None,
                       motion_permission=False,route_selection=False)
        finite=wheel is not None and all(math.isfinite(float(v)) for v in (speed,wheel,yaw,dt))
        if not fresh or not finite or not .01<=dt<=.12 or speed<0:
            self.clear();self.info['reason']='stale_or_invalid_feedback';return None
        self.elapsed+=dt
        self.travel+=(speed if self.previous_speed is None else (speed+self.previous_speed)/2)/3.6*dt
        self.previous_speed=float(speed)
        if turning or abs(wheel)>8 or abs(yaw)>12:
            self.clear();self.info['reason']='turning_or_optical_axis_unstable';return None
        rh,rw=shape[:2]
        if self.shape!=(rh,rw):
            self.samples.clear();self.kind=None;self.shape=(rh,rw)
        candidates=[]
        for d in detections:
            if d['class'] not in ('turn_left','turn_right') or d['confidence']<.65:continue
            if route in ('turn_left','turn_right') and d['class']!=route:continue
            x1,y1,x2,y2=map(float,d['box'])
            if not all(math.isfinite(v) for v in (x1,y1,x2,y2)) or not (2<x1<x2<rw-2 and 2<y1<y2<rh-2):continue
            bw,bh=x2-x1,y2-y1
            if min(bw,bh)<12 or not .67<bw/bh<1.50:continue
            candidates.append((d['class'],min(bw,bh)/rw,((x1+x2)/(2*rw),(y1+y2)/(2*rh))))
        while self.samples and self.elapsed-self.samples[0]['time']>.90:self.samples.popleft()
        if len(candidates)!=1:
            self.info['reason']='missing_clipped_or_competing_faces';return None
        kind,size,center=candidates[0]
        if self.kind!=kind or (self.samples and max(abs(center[i]-self.samples[-1]['center'][i]) for i in (0,1))>.12):
            self.samples.clear();self.kind=kind
        if not self.samples or abs(size-self.samples[-1]['size'])>1e-7:
            self.samples.append(dict(time=self.elapsed,travel=self.travel,size=size,center=center))
        self.info.update(observations=len(self.samples),kind=kind,size_fraction=size)
        if route!=kind:
            self.info['reason']='face_not_yet_voted_for_turn';return None
        if len(self.samples)<3 or self.samples[-1]['time']-self.samples[0]['time']<.25:
            self.info['reason']='insufficient_distinct_current_scale_observations';return None
        scales=[]
        for i,a in enumerate(self.samples):
            for b in list(self.samples)[i+1:]:
                elapsed=b['time']-a['time'];distance=b['travel']-a['travel']
                inverse=1/a['size']-1/b['size']
                if elapsed>=.16 and distance>=.30 and b['size']>=1.04*a['size'] and inverse>0:
                    scales.append(distance/inverse)
        if len(scales)<3:
            self.info['reason']='insufficient_measured_growth_pairs';return None
        scale=statistics.median(scales)
        spread=statistics.median(abs(k-scale) for k in scales)
        if scale<=0 or spread/scale>.25:
            self.info['reason']='inconsistent_optical_scale';return None
        # Use the lower bound of measured distance and reserve latency plus
        # time for wheel alignment. 1.1m/s2 is below the slowest retained
        # straight-brake episode (1.182), not a universal dynamics guarantee.
        v=speed/3.6;entry=18/3.6
        distance=max(0.0,(scale-2*spread)/size-v*.12)
        reserve=max(0.0,(v*v-entry*entry)/(2*1.1))+v*.25+entry*.40
        urgent=speed>19.5 and distance<=reserve
        self.info.update(reason='measured_turn_braking_reserve' if urgent else 'sufficient_current_approach_reserve',
                         applied=urgent,limit=18.0 if urgent else None,
                         distance_lower_m=distance,braking_reserve_m=reserve,
                         scale_m=scale,scale_mad_m=spread,pairs=len(scales),
                         measured_speed_kmh=float(speed),entry_ceiling_kmh=18.0,
                         deceleration_prior_m_s2=1.1,latency_reserve_s=.25,
                         wheel_alignment_reserve_s=.40)
        return 18.0 if urgent else None
