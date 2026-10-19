from speed_profile import band, OFFSET
"""Current selected-turn path owns steering and path-local caps together.

Image corridors are observation proxies, not calibrated physical footprints.
No stored trajectory, checkpoint, world position or map timing is used.
"""
import cv2
import numpy as np
from quality_guard import current_white_mask


class TurnPathContract:
    def __init__(self):
        self.direction = 0
        self.clear_frames = 0
        self.clear_s = 0.0
        self.last_source = None

    def reset(self):
        self.direction = 0
        self.clear_frames = 0
        self.clear_s = 0.0
        self.last_source = None

    def evaluate(self, path, base, raw, *, shape, direction, committed_goal,
                 entry_braking, progress, speed, wheel, yaw, dt, elapsed,
                 legal, hazard):
        info = dict(owns_path=False,cap_scope='unselected_road',
                    reason='no_current_legal_branch',path=None)
        if (path is None or direction not in (-1,1) or not legal or hazard
                or progress>115 or raw is None
                or base.mode in ('status_stale','final_status_stale','manual_assist')):
            self.reset()
            return info
        if wheel is None or not np.isfinite([speed,wheel,yaw,committed_goal,dt]).all():
            self.reset()
            return dict(info,reason='invalid_physical_feedback')
        h,w=shape
        is_arc=path.get('source')=='current_legal_branch_arc_lattice'
        if path.get('source') not in ('current_broad_junction_floor','current_legal_branch_arc_lattice'):
            self.reset()
            return info
        # A clipped outgoing viewport is not the selected arc footprint.
        # Arc already checks every tube radius on current connected Road;
        # keep the absolute anchor margin used by the existing close-curve
        # branch. Broad floor retains its original fractional guard.
        margin_ok=(path.get('image_margin_px',0)>=.025*w and
                   path.get('minimum_tube_clearance_px',-1)>=0) if is_arc else path.get('near_clearance',0)>.20
        if (path.get('direction')!=direction or not margin_ok
                or path.get('foot_road_fraction',0)<.95
                or path.get('current_white_fraction',1)>.12
                or (is_arc and path.get('minimum_tube_clearance_px',-1)<0)):
            self.reset()
            return dict(info,reason='selected_path_lacks_current_clearance')
        white=current_white_mask(cv2.resize(raw,(w,h),interpolation=cv2.INTER_LINEAR))
        if float(white[int(.55*h):].mean())>.80:
            self.reset()
            return dict(info,reason='opaque_current_ground')
        # Check white coverage along THIS current arc, including its far end;
        # foot-only support must not license movement through an opaque patch.
        path_white=float(path['current_white_fraction'])
        if is_arc:
            for row in path['rows']:
                y=min(h-1,max(0,int(row['y']*h)));x=row['center'];r=row['half_width_px']
                patch=white[max(0,y-2):min(h,y+3),max(0,int(x-r)):min(w,int(x+r)+1)]
                if not patch.size:
                    self.reset();return dict(info,reason='unobserved_selected_tube')
                path_white=max(path_white,float(patch.mean()))
        if path_white>.12:
            self.reset()
            return dict(info,reason='selected_tube_occluded')
        turn_goal=direction*float(np.interp(max(0,progress),[0,45,70,90],[25,23,12,0]))
        blend=float(np.clip((max(0,progress)-65)/20,0,1))
        turn_goal=(1-blend)*turn_goal+blend*base.angle
        goal=(base.angle if entry_braking and speed>20 else turn_goal if entry_braking else committed_goal)
        if is_arc and not (entry_braking and speed>20):
            reach=.99-float(path['target_y'])
            goal=float(np.clip(35*(.39/max(.19,reach))*(path['target']-w/2)/(w/2),-25,25))
        goal=float(goal)
        changed=self.last_source is not None and self.last_source!=path['source']
        if self.direction!=direction or changed:
            self.clear_frames=0;self.clear_s=0.0
        self.direction=direction;self.last_source=path['source']
        if path_white<.06:
            self.clear_frames+=1;self.clear_s+=float(np.clip(dt,.01,.25))
        else:
            self.clear_frames=0;self.clear_s=0.0
        selected=dict(path,requested_goal_angle=goal,
                      observation_time_s=float(elapsed),observation_age_s=0.0,
                      current_white_fraction=path_white,
                      steering_source='selected_current_arc_depth_normalized' if is_arc else 'committed_turn_on_current_floor')
        directed=direction*float(wheel)
        error=abs(float(wheel)-goal)
        aligned=(error<=5 and directed>=(4 if is_arc else 18))
        # Reuse the previously measured effective bicycle response to tell
        # intended turn yaw from a departure. It is not a map lookup.
        expected=float(np.degrees((speed/3.6)*np.tan(np.radians(wheel))/2.2125177464858163))
        residual=abs(yaw-expected)
        yaw_ok=(abs(yaw)<35 or (abs(yaw)<65 and direction*yaw>=0 and
                aligned and residual<=8 and speed<=band(23)))
        rolling=(not entry_braking and not aligned and directed>=6 and
                 speed<=10 and abs(yaw)<20 and progress<20)
        cap=band(20.0) if aligned and not entry_braking and speed<=band(23) and yaw_ok else 8.0 if rolling and yaw_ok else 0.0
        reason=('selected_path_tracking' if cap==band(20) else 'selected_path_rolling' if cap else
                'entry_momentum_brake' if entry_braking else 'selected_path_overspeed' if speed>band(23) else
                'selected_path_yaw_mismatch' if not yaw_ok else 'wheel_not_on_selected_path')
        # No remembered white warning from another path can import its
        # unrelated settle/edge cap. Retain the existing 8/11 water band until
        # this current selected corridor is dry in two timed observations.
        dry_confirmed=self.clear_frames>=2 and self.clear_s>=.16
        if path_white>=.06 or (base.spray_prealert and not dry_confirmed):
            cap=min(cap,8.0) if speed<=11 else 0.0
            reason='selected_path_white_caution'
        mode='turn_junction_tracking' if cap==band(20) else 'turn_junction_rolling' if cap>0 else 'turn_junction_align'
        return dict(owns_path=True,path=selected,goal_angle=goal,speed_cap=cap,mode=mode,
                    cap_scope='selected_current_turn',cap_reason=reason,reason='current_legal_branch_observed',
                    source_transition=changed,continuity='current_observation_and_physical_goal_agreement',
                    wheel_error_deg=float(error),actual_wheel=float(wheel),actual_speed=float(speed),
                    actual_yaw_deg_s=float(yaw),expected_yaw_deg_s=expected,yaw_residual_deg_s=float(residual),
                    yaw_motion_ok=bool(yaw_ok),selected_white_fraction=path_white,dry_confirmed=dry_confirmed,
                    road_cap_not_selected=dict(mode=base.mode,speed=float(base.speed),angle=float(base.angle)),
                    speed_band_unchanged=OFFSET==0)
