"""Use one current outgoing Road proposal for steering and its safety cap.

The Road proposal is already filtered and checked against measured wheels.
An additional Traffic slew must not retain the old turn after the selected
outgoing corridor has changed. This contract never raises a speed cap.
"""
import math
import cv2
import numpy as np
from quality_guard import current_white_mask


def current_outgoing_contract(base, road, raw, *, mode, goal, previous_command,
                              wheel, speed, yaw, elapsed, hazard):
    info = dict(applied=False, reason='not_outgoing_road', path=None)
    if mode not in ('turn_exit', 'turn_handover') or hazard:
        return info
    if raw is None or wheel is None or not all(math.isfinite(float(v)) for v in
            (goal, previous_command, wheel, speed, yaw, base.angle, elapsed)):
        return dict(info, reason='invalid_current_feedback')
    if (base.confidence != 1 or base.preview_quality < .6
            or base.near_clearance <= .28 or base.occlusion_fraction >= .12
            or base.junction_suspect or base.junction_hold_path
            or base.geometry_guard or base.edge_trend_guard or base.mask_shift_guard
            or base.mode in ('status_stale', 'final_status_stale', 'manual_assist')
            or abs(goal-base.angle) > 1e-9):
        return dict(info, reason='outgoing_road_not_resolved')
    # The change is unloading an old turn request. It never authorizes a
    # faster build-up of steering amplitude or a large sign reversal.
    if abs(goal) > 6 or abs(goal) >= abs(previous_command) or abs(goal) > abs(wheel):
        return dict(info, reason='not_unwinding_to_current_road')
    h, w = road.shape
    centers, widths = base.centers, base.widths
    if any(v is None or not math.isfinite(float(v)) for v in centers+widths):
        return dict(info, reason='missing_current_rows')
    if not (widths[0] > 1.10*widths[1] and widths[1] > 1.25*widths[2]
            and .08*w < widths[2] < .70*widths[1]
            and centers[1]-widths[1]/2 > .03*w
            and centers[1]+widths[1]/2 < .97*w):
        return dict(info, reason='outgoing_perspective_not_resolved')
    white = current_white_mask(cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR))
    foot = road[int(.88*h):int(.97*h), int(.43*w):int(.57*w)]
    if not foot.size or float(foot.mean()) < .95:
        return dict(info, reason='missing_current_foot')
    # Verify a continuous image tube from the car region to the observed
    # far center. This is image support, not a calibrated swept footprint.
    ys = np.array([.55, .70, .85, .95])*h
    xs = np.array([centers[2], centers[1], centers[0], w/2])
    spans = np.array([widths[2], widths[1], widths[0], widths[0]])
    support = 1.0
    whiteness = 0.0
    for y in range(int(ys[0]), int(ys[-1])+1, 2):
        x = float(np.interp(y, ys, xs))
        radius = .02*w
        a, b = int(x-radius), int(x+radius)+1
        if a < 0 or b > w:
            return dict(info, reason='current_tube_leaves_view')
        patch = road[max(0,y-1):min(h,y+2), a:b]
        # Brightness over a marking-width tube confuses the visible painted
        # center line with opaque spray. Measure the CURRENT inner corridor
        # around this same center; never fill a missing mask along the tube.
        wr = .30*float(np.interp(y,ys,spans))
        wa, wb = max(0,int(x-wr)), min(w,int(x+wr)+1)
        wp = white[max(0,y-1):min(h,y+2), wa:wb]
        support = min(support, float(patch.mean()))
        whiteness = max(whiteness, float(wp.mean()))
    if support < .85 or whiteness > .12:
        return dict(info, reason='current_tube_missing_or_occluded',
                    minimum_support=support, maximum_white=whiteness)
    path = dict(source='current_outgoing_road', observation_time_s=float(elapsed),
        observation_age_s=0.0, rows=[dict(y=y, center=float(c), width=float(v))
            for y,c,v in zip((.85,.70,.55),centers,widths)],
        near_bounds=[base.near_left,base.near_right],
        foot_road_fraction=float(foot.mean()), minimum_tube_support=support,
        maximum_inner_corridor_white=whiteness, requested_goal_angle=float(base.angle),
        current_observation=True, physical_footprint_calibrated=False)
    return dict(applied=True, reason='one_steering_owner_on_current_outgoing_road',
        path=path, goal_angle=float(base.angle), road_cap=float(base.speed),
        previous_traffic_command=float(previous_command), actual_wheel=float(wheel),
        actual_speed=float(speed), actual_yaw=float(yaw),
        steering_source='road_filtered_command', cap_source='same_road_physical_guards',
        raises_motion_permission=False)
