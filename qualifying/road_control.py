"""Road-mask controller for the UCR 2026 camera. No BTC library calls here."""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class Decision:
    speed: float
    angle: float
    target_x: float
    confidence: float
    centers: tuple
    widths: tuple
    curvature: float
    near_left: float | None
    near_right: float | None
    mode: str = 'normal'
    preview_bend: float = 0.0
    junction_suspect: bool = False
    yaw_rate: float = 0.0
    near_clearance: float = 0.0
    bias_applied: float = 0.0
    preview_quality: float = 0.0
    junction_hold_path: bool = False
    junction_heading_error: float = 0.0
    geometry_guard: bool = False
    edge_trend_guard: bool = False
    junction_forward_path: bool = False
    mode_transition_guard: bool = False
    geometry_target_hold: bool = False
    clipped_near_preview: bool = False
    geometric_derivative: float = 0.0
    overspeed_caution: bool = False

class RoadController:

    def __init__(self, *, steer_sign=1.0, right_bias=0.0, min_speed=12.0, max_speed=25.0):
        self.steer_sign = float(steer_sign)
        self.right_bias = float(right_bias)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.last_geometric_error = None
        self.error_rate = 0.0
        self.last_angle = 0.0
        self.last_target = None
        self.last_speed_cmd = self.min_speed
        self.elapsed_s = 0.0
        self.frames = 0
        self.straight_s = 0.0
        self.hazard_hold_s = 0.0
        self.junction_hold_s = 0.0
        self.bias_applied = 0.0
        self.last_lane_error = 0.0
        self.last_lane_near_width = None
        self.junction_near_width = None
        self.last_lane_mid_width = None
        self.junction_mid_width = None
        self.junction_entry_error = 0.0
        self.junction_heading_error = 0.0
        self.was_junction = False
        self.junction_age_s = 0.0
        self.last_geometry_center = None
        self.last_geometry_width = None
        self.last_geometry_bounds = None
        self.pending_geometry_center = None
        self.pending_geometry_count = 0
        self.geometry_reject_s = 0.0
        self.last_margins = None
        self.margin_rates = np.zeros(2, dtype=float)
        self.last_resolved_curvature = 0.0
        self.preview_missing_s = 0.0
        self.last_path_mode = None
        self.transition_remaining_s = 0.0
        self.visibility_pulse_s = 0.0
        self.visibility_cooldown_s = 0.0

    @staticmethod
    def _runs(row):
        edges = np.diff(np.r_[False, row, False].astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1) - 1
        return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= 7]

    @staticmethod
    def _road_slice(road, y):
        radius = max(2, int(round(0.0185 * road.shape[0])))
        band = road[max(0, y - radius):min(road.shape[0], y + radius + 1)]
        return band.sum(axis=0) > len(band) // 2

    def decide(self, mask, dt_s=0.083, actual_speed=0.0, yaw_rate=0.0):
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError('Road mask must be a 2D uint8 array')
        dt_s = float(np.clip(dt_s, 0.01, 0.25))
        self.elapsed_s += dt_s
        self.visibility_pulse_s = max(0.0, self.visibility_pulse_s - dt_s)
        self.visibility_cooldown_s = max(0.0, self.visibility_cooldown_s - dt_s)
        h, w = mask.shape
        road = cv2.morphologyEx((mask == 1).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        if road.mean() > 0.9 and road[:max(1, int(0.25 * h))].mean() > 0.8:
            return self._lost(w, dt_s, reason='unstructured_mask')
        n, labels, stats, _ = cv2.connectedComponentsWithStats(road)
        if n <= 1:
            return self._lost(w, dt_s)
        anchor = labels[int(h * 0.9):, int(w * 0.28):int(w * 0.72)]
        counts = np.bincount(anchor.ravel(), minlength=n)
        counts[0] = 0
        component = int(np.argmax(counts))
        if counts[component] < 25:
            return self._lost(w, dt_s)
        road = labels == component
        centers = []
        widths = []
        bounds = []
        run_counts = []
        previous = w / 2
        for fraction in (0.85, 0.7, 0.55):
            y = min(h - 2, max(1, int(fraction * h)))
            row = self._road_slice(road, y)
            runs = self._runs(row)
            run_counts.append(len(runs))
            if not runs:
                centers.append(None)
                widths.append(None)
                bounds.append(None)
                continue
            a, b = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2 - previous) - 0.15 * (ab[1] - ab[0]))
            previous = (a + b) / 2
            centers.append(previous)
            widths.append(float(b - a + 1))
            bounds.append((float(a), float(b)))
        valid = [i for i, c in enumerate(centers) if c is not None]
        if len(valid) < 2:
            return self._lost(w, dt_s)
        near = centers[0] if centers[0] is not None else centers[valid[0]]
        middle = centers[1] if centers[1] is not None else near
        far = centers[2] if centers[2] is not None else middle
        near_width = widths[0] if widths[0] is not None else widths[valid[0]]
        center = 0.65 * near + 0.35 * middle
        curvature = (far - middle) / (w / 2)
        near_slope = (middle - near) / (w / 2)
        preview_quality = min(1.0, widths[2] / (0.15 * w)) if widths[2] else 0.0
        if bounds[2] and (bounds[2][0] <= 1 or bounds[2][1] >= w - 2):
            preview_quality *= 0.5
        curvature *= max(0.35, preview_quality)
        if centers[2] is None:
            self.preview_missing_s += dt_s
            if self.preview_missing_s <= 0.18:
                curvature = self.last_resolved_curvature * max(0.0, 1.0 - self.preview_missing_s / 0.25)
        else:
            self.preview_missing_s = 0.0
        far_bounds = bounds[2]
        expanded_far = far_bounds is not None and widths[1] is not None and (widths[2] > 0.9 * widths[1]) and (widths[2] > 0.45 * w) and (far_bounds[0] > 0.03 * w and far_bounds[1] < 0.97 * w or widths[2] > 0.85 * w or (abs(near_slope) < 0.12 and widths[2] > 1.18 * widths[1]))
        preview_bend = abs(curvature)
        branched = False
        for fraction in (0.5, 0.46):
            y = min(h - 2, max(1, int(fraction * h)))
            runs = self._runs(self._road_slice(road, y))
            if not runs:
                continue
            large_runs = [(a, b) for a, b in runs if b - a + 1 > 0.04 * w]
            branched |= len(large_runs) > 1 and large_runs[-1][0] - large_runs[0][1] > 0.1 * w
            a, b = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2 - far))
            if b - a + 1 >= 0.08 * w and a > 0 and (b < w - 1):
                extra_bend = ((a + b) / 2 - far) / (w / 2)
                consistent = abs(curvature) > 0.04 and extra_bend * curvature > 0
                if not consistent:
                    continue
                preview_bend = max(preview_bend, min(abs(extra_bend), 0.5) * 0.6)
        near_crossing = self.was_junction and self.junction_near_width is not None and (widths[0] is not None) and (widths[0] > 1.15 * self.junction_near_width)
        mid_crossing = self.was_junction and widths[1] is not None and (widths[1] > 0.97 * near_width or (self.junction_mid_width is not None and widths[1] > 1.25 * self.junction_mid_width))
        if expanded_far or branched or near_crossing or mid_crossing:
            self.junction_hold_s = 0.65
        else:
            self.junction_hold_s = max(0.0, self.junction_hold_s - dt_s)
        junction = self.junction_hold_s > 0.0
        hold_path = False
        forward_path = False
        self.junction_age_s = self.junction_age_s + dt_s if junction else 0.0
        if junction:
            if not self.was_junction:
                self.junction_entry_error = self.last_lane_error
                self.junction_heading_error = 0.0
                self.junction_near_width = self.last_lane_near_width
                self.junction_mid_width = self.last_lane_mid_width
            self.junction_heading_error += float(yaw_rate) * dt_s
            middle_expanded = widths[1] is not None and widths[1] > 0.97 * near_width
            center = near if middle_expanded else 0.85 * near + 0.15 * middle
            if middle_expanded:
                near_slope = 0.0
            hold_path = widths[1] is not None and (middle_expanded or mid_crossing or (near_width > 0.9 * w and widths[1] > 0.8 * w))
            if hold_path:
                center = w / 2 + self.junction_entry_error * w / 2
                near_slope = 0.0
            elif near_crossing:
                center = middle
                near_slope = 0.0
            if self.junction_age_s > 2.5 and run_counts[2] == 1 and (widths[2] is not None) and (widths[1] is not None) and (widths[2] < 0.75 * widths[1]) and (far_bounds is not None) and (far_bounds[0] > 0.03 * w) and (far_bounds[1] < 0.97 * w) and (not branched):
                center = far
                near_slope = 0.0
                hold_path = False
                forward_path = True
            curvature = 0.0
        else:
            self.junction_heading_error = 0.0
            if centers[2] is not None:
                self.last_resolved_curvature = curvature
        if junction:
            self.last_resolved_curvature = 0.0
            self.preview_missing_s = 0.0
        self.was_junction = junction
        near_bounds = bounds[0] if bounds[0] is not None else bounds[valid[0]]
        observed_bounds = near_bounds
        observed_width = near_width
        observed_clearance = min(w / 2 - observed_bounds[0], observed_bounds[1] - w / 2) / max(observed_width, 1.0)
        geometry_guard = False
        if self.last_geometry_center is not None and self.last_geometry_width:
            previous_clearance = min(w / 2 - self.last_geometry_bounds[0], self.last_geometry_bounds[1] - w / 2) / self.last_geometry_width
            width_ratio = min(near_width, self.last_geometry_width) / max(near_width, self.last_geometry_width)
            suspect = not junction and abs(yaw_rate) < 25.0 and (abs(center - self.last_geometry_center) > 0.08 * w) and (width_ratio < 0.8 or len(valid) < 3)
            if suspect:
                if self.pending_geometry_center is not None and abs(center - self.pending_geometry_center) < 0.04 * w:
                    self.pending_geometry_count += 1
                else:
                    self.pending_geometry_count = 1
                self.pending_geometry_center = center
                self.geometry_reject_s += dt_s
                geometry_guard = self.pending_geometry_count < 2 and self.geometry_reject_s <= 0.25 and (observed_clearance > 0.32) and (previous_clearance > 0.32)
        geometry_target_hold = geometry_guard and self.last_target is not None and (observed_bounds[0] + 0.2 * observed_width < self.last_target < observed_bounds[1] - 0.2 * observed_width)
        if geometry_guard:
            center = self.last_geometry_center
            near_width = self.last_geometry_width
            near_bounds = self.last_geometry_bounds
            curvature = near_slope = preview_bend = 0.0
        else:
            self.last_geometry_center = center
            self.last_geometry_width = near_width
            self.last_geometry_bounds = near_bounds
            self.geometry_reject_s = 0.0
            self.pending_geometry_center = None
            self.pending_geometry_count = 0
        if not junction and (not geometry_guard):
            self.last_lane_error = (center - w / 2) / (w / 2)
            if len(valid) == 3:
                if self.last_lane_near_width is None:
                    self.last_lane_near_width = near_width
                    self.last_lane_mid_width = widths[1]
                else:
                    a = dt_s / (0.5 + dt_s)
                    self.last_lane_near_width += a * (near_width - self.last_lane_near_width)
                    self.last_lane_mid_width += a * (widths[1] - self.last_lane_mid_width)
        left_margin = w / 2 - near_bounds[0]
        right_margin = near_bounds[1] - w / 2
        clearance = min(left_margin, right_margin) / max(near_width, 1.0)
        margins = np.array((left_margin, right_margin), dtype=float) / w
        if self.last_margins is not None:
            rates = np.clip((margins - self.last_margins) / dt_s, -4.0, 4.0)
            self.margin_rates += dt_s / (0.15 + dt_s) * (rates - self.margin_rates)
        self.last_margins = margins
        predicted_margins = margins + 0.35 * self.margin_rates
        approaching_edge = bool(np.any((predicted_margins < 0.09) & (self.margin_rates < -0.08)))
        edge_guard = min(left_margin, right_margin) <= 0 or (clearance < 0.33 and approaching_edge and (actual_speed > 8))
        geometry_target_hold = geometry_target_hold and (not edge_guard)
        path_mode = (junction, hold_path, forward_path)
        if self.last_path_mode is not None and path_mode != self.last_path_mode:
            self.transition_remaining_s = 0.22
        self.last_path_mode = path_mode
        mode_transition_guard = self.transition_remaining_s > 0 and clearance > 0.32 and (len(valid) >= 2) and (abs(yaw_rate) < 35) and (not edge_guard) and (not geometry_guard)
        self.transition_remaining_s = max(0.0, self.transition_remaining_s - dt_s)
        clipped_near_preview = not junction and bounds[0] is not None and (bounds[0][0] <= 1) and (bounds[0][1] >= w - 2) and (len(valid) == 3) and (not geometry_guard) and (run_counts[1] == 1) and (run_counts[2] == 1) and (widths[1] < 0.85 * widths[0]) and (widths[2] < 0.75 * widths[1]) and (abs(near_slope) > 0.28) and ((far - near) * (middle - near) > 0)
        bias_safety = float(np.clip((clearance - 0.24) / 0.16, 0.0, 1.0))
        bias_safety *= float(np.clip((0.4 - preview_bend) / 0.25, 0.0, 1.0))
        if len(valid) < 3 or junction or geometry_guard or edge_guard or clipped_near_preview or (abs(yaw_rate) > 35.0):
            bias_safety = 0.0
        desired_bias = max(self.right_bias, 0.0) * bias_safety
        if geometry_target_hold:
            desired_bias = self.bias_applied
        if desired_bias < self.bias_applied:
            immediate = clearance < 0.35 or edge_guard or geometry_guard or hold_path or (abs(yaw_rate) > 25.0)
            if immediate:
                self.bias_applied = desired_bias
            else:
                self.bias_applied += dt_s / (0.12 + dt_s) * (desired_bias - self.bias_applied)
                if abs(self.bias_applied - desired_bias) < 0.0001:
                    self.bias_applied = desired_bias
        else:
            self.bias_applied += dt_s / (0.4 + dt_s) * (desired_bias - self.bias_applied)
        offset = min(self.bias_applied * near_width, 0.1 * w)
        target = float(np.clip(center + offset, 0.12 * w, 0.88 * w))
        error = (target - w / 2) / (w / 2)
        geometric_error = (center - w / 2) / (w / 2)
        raw_rate = 0.0 if self.last_geometric_error is None else float(np.clip((geometric_error - self.last_geometric_error) / dt_s, -4.0, 4.0))
        target_jump = self.last_target is not None and abs(target - self.last_target) > 0.12 * w
        derivative_alpha = dt_s / (0.1 + dt_s)
        self.error_rate += derivative_alpha * (raw_rate - self.error_rate)
        derivative = float(np.clip(1.5 * self.error_rate, -3.0, 3.0))
        desired_angle = 14.0 * error + 3.0 * curvature + 2.0 * near_slope + derivative
        if hold_path:
            desired_angle -= float(np.clip(0.35 * self.junction_heading_error, -5.0, 5.0))
        safe_margin = max(0.1 * w, 0.26 * near_width)
        if left_margin < safe_margin:
            desired_angle += 10.0 * (safe_margin - left_margin) / safe_margin
        if right_margin < safe_margin:
            desired_angle -= 10.0 * (safe_margin - right_margin) / safe_margin
        steer_desired = desired_angle + 2.0 * error + 0.5 * curvature
        damping = 0.1 if junction else 0.06
        steer_desired -= float(np.clip(damping * yaw_rate, -5.0, 5.0))
        alpha = 1.0 - 0.35 ** (dt_s / 0.083)
        steering_limit = float(np.interp(float(actual_speed), (15.0, 30.0), (22.0, 16.0)))
        angle = float(np.clip((1.0 - alpha) * self.last_angle + alpha * steer_desired, -steering_limit, steering_limit))
        if mode_transition_guard:
            angle = float(np.clip(angle, self.last_angle - 18.0 * dt_s, self.last_angle + 18.0 * dt_s))
        turn = min(1.0, abs(desired_angle) / 13.5 + preview_bend * 1.8)
        speed = self.max_speed - (self.max_speed - self.min_speed) * turn
        mode = 'normal'
        reversing_with_inertia = self.last_angle * steer_desired < 0.0 and abs(self.last_angle) > 5.0 and (abs(yaw_rate) > 35.0)
        hazard = target_jump or abs(yaw_rate) > 65.0 or reversing_with_inertia or (clearance < 0.18)
        if hazard:
            self.hazard_hold_s = 0.5
        else:
            self.hazard_hold_s = max(0.0, self.hazard_hold_s - dt_s)
        if preview_bend > 0.25:
            speed = min(speed, 18.0)
            mode = 'corner'
        if preview_bend > 0.45:
            speed = min(speed, self.min_speed)
        if abs(yaw_rate) > 35.0:
            speed = min(speed, 18.0)
            mode = 'recovery'
        if self.hazard_hold_s > 0.0:
            speed = min(speed, 12.0)
            mode = 'recovery'
        if junction:
            speed = min(speed, 14.0)
            mode = 'junction_suspect'
        if forward_path:
            speed = min(speed, 10.0)
            mode = 'junction_forward'
        if clipped_near_preview:
            speed = min(speed, 14.0)
            mode = 'preview_uncertain'
        if min(left_margin, right_margin) < max(0.07 * w, 0.2 * near_width):
            speed = min(speed, self.min_speed)
            mode = 'recovery'
        confidence = len(valid) / 3
        if confidence < 1.0:
            speed = min(speed, self.min_speed)
        if geometry_guard:
            speed = min(speed, 8.0)
            mode = 'perception_guard'
        if mode_transition_guard:
            speed = min(speed, 10.0)
            mode = 'mode_transition'
        if edge_guard:
            speed = min(speed, 8.0)
            mode = 'edge_trend_guard'
        visibility_risk = preview_quality < 0.1 and 0.26 < clearance < 0.45 and (abs(yaw_rate) > 7.0) and (actual_speed > 18.0) and (not junction) and (not geometry_guard) and (not edge_guard)
        edge_visibility_risk = preview_quality < 0.6 and clearance < 0.24 and (abs(yaw_rate) > 10.0) and (actual_speed > 13.0) and (not junction) and (not geometry_guard) and (not edge_guard)
        if visibility_risk and self.visibility_cooldown_s <= 0.0 or (edge_visibility_risk and self.visibility_cooldown_s <= 0.25 and (self.visibility_pulse_s <= 0.0)):
            self.visibility_pulse_s = 0.24
            self.visibility_cooldown_s = 2.3
        if self.visibility_pulse_s > 0.0:
            speed = 0.0
            mode = 'visibility_caution'
        if min(left_margin, right_margin) <= 0:
            speed = 0.0
        if self.elapsed_s < 3.32:
            speed = min(speed, self.min_speed + self.elapsed_s * 3.0)
        momentum_risk = preview_bend > 0.25 or abs(near_slope) > 0.25 or junction or clipped_near_preview or geometry_guard or edge_guard or mode_transition_guard or (confidence < 1) or (self.hazard_hold_s > 0)
        tolerance = 2.0 if momentum_risk else 5.0
        overspeed = max(0.0, float(actual_speed) - speed - tolerance)
        overspeed_caution = bool(momentum_risk and overspeed > 0)
        speed = max(0.0, speed - (0.8 if momentum_risk else 0.4) * overspeed)
        speed = min(speed, self.last_speed_cmd + 12.0 * dt_s)
        self.last_geometric_error = geometric_error
        self.last_angle = angle
        self.last_target = target
        self.last_speed_cmd = speed
        self.frames += 1
        return Decision(float(speed), float(self.steer_sign * angle), target, confidence, tuple(centers), tuple(widths), float(curvature), *near_bounds, mode, float(preview_bend), bool(junction), float(yaw_rate), float(clearance), float(self.bias_applied), float(preview_quality), bool(hold_path), float(self.junction_heading_error), bool(geometry_guard), bool(edge_guard), bool(forward_path), bool(mode_transition_guard), bool(geometry_target_hold), bool(clipped_near_preview), float(derivative), bool(overspeed_caution))

    def _lost(self, w, dt_s=0.083, reason='mask_lost'):
        self.frames += 1
        self.straight_s = 0.0
        self.last_angle *= 0.8 ** (dt_s / 0.083)
        self.error_rate = 0.0
        self.last_geometric_error = None
        self.last_speed_cmd = 0.0
        self.hazard_hold_s = 0.5
        self.bias_applied = 0.0
        self.was_junction = False
        self.junction_hold_s = 0.0
        self.junction_heading_error = 0.0
        self.junction_age_s = 0.0
        self.last_geometry_center = None
        self.last_geometry_width = None
        self.last_geometry_bounds = None
        self.pending_geometry_center = None
        self.pending_geometry_count = 0
        self.geometry_reject_s = 0.0
        self.last_margins = None
        self.margin_rates[:] = 0
        self.last_resolved_curvature = 0.0
        self.preview_missing_s = 0.0
        self.last_path_mode = None
        self.transition_remaining_s = 0.0
        return Decision(0.0, float(self.steer_sign * self.last_angle), self.last_target if self.last_target is not None else w / 2, 0.0, (None, None, None), (None, None, None), 0.0, None, None, reason)
