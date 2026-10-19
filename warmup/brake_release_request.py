"""Respect BTC's retained brake while resuming a currently safe road request.

BTC only clears brakeTorque when its positive command is above rounded
measured speed. A rate-limited command below speed can keep braking even
after the current path no longer needs deceleration. This function changes
the emitted request only within the already computed physical ceiling.
"""
import math


class BrakeReleaseRequest:
    def __init__(self):
        self.clear_s = 0.0
        self.info = {}
        self.evaluated = False

    def begin_observation(self):
        if not self.evaluated:
            self.clear_s = 0.0
        self.evaluated = False
        self.info = dict(applied=False,reason='not_evaluated_on_current_frame')

    def apply(self, limited, ceiling, *, dt, actual_speed, wheel, angle,
              yaw, eligible, curve_release=False):
        self.evaluated = True
        values=(limited,ceiling,dt,actual_speed,wheel,angle,yaw)
        if wheel is None or not all(math.isfinite(float(v)) for v in values):
            self.clear_s=0.0
            self.info=dict(applied=False,reason='invalid_physical_feedback')
            return limited
        # The existing speed ceiling is authoritative. The new request can
        # never clear an actual overspeed brake or a current hard stop.
        expected_yaw=math.degrees((actual_speed/3.6)*math.tan(math.radians(wheel))/2.2125177464858163)
        yaw_ready=(abs(yaw)<12.0 or (curve_release and abs(yaw)<35.0 and abs(yaw-expected_yaw)<8.0))
        ready=(eligible and ceiling>0 and 0<=actual_speed<ceiling-1.0
               and abs(wheel-angle)<3.0 and yaw_ready and 0<dt<=.12)
        self.clear_s=self.clear_s+dt if ready else 0.0
        request=limited
        applied=False
        reason='current_physical_or_path_veto' if not ready else 'confirm_current_release'
        if self.clear_s>=.12:
            # 1.0 km/h exceeds the rounding uncertainty by at least0.5.
            # There is still no extra ceiling or permission to roll through
            # a hidden path; downstream Traffic/Recovery/stale caps remain.
            floor=min(ceiling,actual_speed+1.0)
            if limited<floor:
                request=floor;applied=True;reason='release_retained_brake_within_current_ceiling'
            else:reason='ordinary_ramp_already_above_actual'
        self.info=dict(applied=applied,reason=reason,confirmation_s=self.clear_s,
            previous_limited_request=float(limited),current_physical_ceiling=float(ceiling),
            actual_speed=float(actual_speed),actual_wheel=float(wheel),
            selected_angle=float(angle),actual_yaw=float(yaw),final_request=float(request),
            changed_ceiling=False)
        return request
