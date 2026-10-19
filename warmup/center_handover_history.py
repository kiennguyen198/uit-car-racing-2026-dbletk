"""Road resumes from the command queued during a verified turn handover.

Never grants motion permission or changes this frame's final command.
Only command history is updated; current Road measurement/edges stay intact.
"""
import math

def adopt_center_handover_history(road, decision, base, traffic, *, fresh):
    info=dict(applied=False,reason='not_verified_turn_handover')
    if traffic.get('mode') not in ('turn_exit','turn_handover'):
        return info
    if not fresh or decision.mode not in ('turn_exit','turn_handover'):
        return dict(info,reason='final_stop_or_stale_not_handover')
    if (traffic.get('turn_direction') or traffic.get('sign_conflicts') or
            traffic.get('obstacles_on_road') or traffic.get('uncertain_side_contacts')):
        return dict(info,reason='traffic_hazard_or_other_owner')
    if (base.confidence!=1 or base.preview_quality<.6 or base.near_clearance<=.28
            or base.junction_suspect or base.geometry_guard or base.edge_trend_guard):
        return dict(info,reason='current_outgoing_road_not_resolved')
    values=(decision.speed,decision.angle,decision.target_x,base.near_left,base.near_right)
    if any(v is None or not math.isfinite(v) for v in values):
        return dict(info,reason='invalid_current_observation_or_command')
    before=dict(angle=road.last_angle*road.steer_sign,speed=road.last_speed_cmd,target=road.last_target)
    road.last_angle=float(decision.angle)/road.steer_sign
    road.last_speed_cmd=float(decision.speed)
    road.last_target=float(decision.target_x)
    road.last_geometric_error=None
    road.error_rate=0.
    return dict(applied=True,reason='verified_outgoing_road_after_turn',
                source='final_queued_command_after_stale_and_hazard_checks',
                before=before,after=dict(angle=decision.angle,speed=decision.speed,target=decision.target_x),
                current_near_bounds=[base.near_left,base.near_right],
                updates_motion_permission=False)
