"""Reach a repeatedly observed turn goal while the existing entry brake is on.

No motion permission, speed cap, turn goal or route selection is changed.
The actual-wheel and emitted-command gates remain in Traffic afterwards.
"""
import math

class TurnAlignmentRequest:
    def __init__(self):
        self.source=None
        self.goal=None
        self.confirm_s=0.0
        self.info={}
        self.evaluated=False
        self.pending_direction=0

    def begin_observation(self):
        if not self.evaluated:
            self.source=None
            self.goal=None
            self.confirm_s=0.0
        self.evaluated=False
        self.info=dict(applied=False,reason='not_evaluated_on_current_frame')

    def rate(self, ordinary, *, contract, entry_braking, direction, speed,
             wheel, goal, dt, cap):
        path=contract.get('path') or {}
        self.evaluated=True
        values=(speed,wheel,goal,dt,cap)
        valid=(wheel is not None and all(math.isfinite(float(x)) for x in values))
        eligible=(valid and contract.get('owns_path') and entry_braking
            and cap==0 and direction in (-1,1) and 10<speed<=20
            and .01<=dt<=.12 and 15<=direction*goal<=25
            and path.get('direction')==direction
            and path.get('observation_age_s',1)==0
            and path.get('current_white_fraction',1)<.06
            and path.get('foot_road_fraction',0)>=.95
            and path.get('source') in ('current_broad_junction_floor',
                                      'current_legal_branch_arc_lattice'))
        source=(path.get('source'),direction)
        same=(eligible and source==self.source and self.goal is not None
              and abs(goal-self.goal)<=2)
        self.confirm_s=(self.confirm_s+dt if same else dt) if eligible else 0.0
        self.source=source if eligible else None
        self.goal=float(goal) if eligible else None
        applied=bool(eligible and self.confirm_s>=.16 and abs(wheel-goal)>5)
        if applied:self.pending_direction=direction
        rate=60.0 if applied else ordinary
        self.info=dict(applied=applied,reason='confirmed_current_goal_during_entry_brake'
            if applied else 'ordinary_slew',confirmation_s=self.confirm_s,
            requested_goal=goal,actual_wheel=wheel,actual_speed=speed,
            selected_path_source=path.get('source'),rate_deg_s=rate,
            unchanged_speed_cap=cap,motion_permission=False)
        return rate

    def physical_cap(self, cap, *, contract, direction, wheel, goal):
        # Faster requested steering must not open the old rolling8 phase
        # while the measured wheel still points down the previous corridor.
        # This carries provenance only, never remembered road geometry.
        if not self.pending_direction:return cap
        if direction!=self.pending_direction:
            self.pending_direction=0
            return cap
        finite=(wheel is not None and all(math.isfinite(float(x)) for x in (wheel,goal)))
        aligned=bool(finite and contract.get('owns_path') and abs(wheel-goal)<=5)
        if aligned:
            self.pending_direction=0
            self.info['physical_alignment_confirmed']=True
            return cap
        self.info.update(physical_alignment_confirmed=False,
            pending_direction=self.pending_direction,physical_cap_veto=True)
        return 0.0
