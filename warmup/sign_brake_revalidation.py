"""Retire an obsolete sign brake only after a current legal path is observed.

The input ceiling is an existing path/approach ceiling, not a new speed band.
Required turns, conflicts and any unresolved/current hazard retain braking.
"""
import math


class SignBrakeRevalidation:
    def __init__(self):
        self.clear_s=0.0
        self.route=None
        self.info={}

    def evaluate(self,old_limit,*,limit,request,speed,wheel,angle,yaw,dt,
                 route,eligible,path_source):
        before=old_limit
        values=(limit,request,speed,wheel,angle,yaw,dt)
        ready=(eligible and old_limit is not None
            and all(v is not None and math.isfinite(float(v)) for v in values)
            and limit>old_limit+.5 and 0<=speed<min(request,limit)-1.0
            and abs(wheel-angle)<=3 and abs(yaw)<8 and .01<=dt<=.12)
        if not ready or route!=self.route:
            self.clear_s=0.0
        if ready:self.clear_s+=dt
        self.route=route if ready else None
        released=ready and self.clear_s>=.16
        self.info=dict(released=bool(released),old_limit=before,
            current_limit=limit,current_request=request,actual_speed=speed,
            actual_wheel=wheel,selected_angle=angle,actual_yaw=yaw,
            selected_route=route,path_source=path_source,
            confirmation_s=self.clear_s,reason=('current_legal_path_revalidates_higher_existing_limit'
                if released else 'old_brake_retained_or_already_inactive'),
            changes_existing_ceiling=False)
        return None if released else old_limit
