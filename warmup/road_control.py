from speed_profile import band
'Road-mask controller for the UCR 2026 camera. No BTC library calls here.'
from dataclasses import dataclass, replace
from collections import deque
import cv2
import numpy as np
from brake_release_request import BrakeReleaseRequest
from quality_guard import visible_spray_corridor, current_near_water_path, current_near_curve_path, fill_enclosed_paint_holes

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
    mask_shift_guard: bool = False
    boundary_recovery_guard: bool = False
    steering_settle_guard: bool = False
    occlusion_fraction: float = 0.0
    occlusion_hold: bool = False
    mask_gap_envelope: bool = False
    spray_prealert: bool = False
    current_path: dict | None = None
from clipped_reference import current_clipped_reference

class RoadController:

    def __init__(self, *, steer_sign=1.0, right_bias=0.0, min_speed=12.0, max_speed=26.0):
        self.steer_sign = float(steer_sign)
        self.right_bias = float(right_bias)
        self.min_speed = float(min_speed)
        self.max_speed = band(float(max_speed))
        self.last_geometric_error = None
        self.error_rate = 0.0
        self.last_angle = 0.0
        self.last_target = None
        self.last_speed_cmd = self.min_speed
        self.brake_release_request = BrakeReleaseRequest()
        self.curve_speed_braking = False
        self.curve_speed_brake_ceiling = None
        self.current_cap_revalidation_s = 0.0
        self.curve_speed_control_info = {}
        self.clipped_reference_info = {}
        self.clipped_reference_previous = None
        self.clipped_reference_confirm_s = 0.0
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
        self.last_geometry_centers = None
        self.last_geometry_widths = None
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
        self.last_visible_mask = None
        self.last_visible_mask_s = -1000000000.0
        self.last_paint_holes = 0
        self.occlusion_elapsed_s = 0.0
        self.occlusion_release_s = 0.0
        self.partial_center = None
        self.partial_visible_frames = 0
        self.spray_center = None
        self.spray_visible_frames = 0
        self.spray_observations = deque()
        self.water_recovery_s = 0.0
        self.water_recovery_m = 0.0
        self.water_clear_s = 0.0
        self.water_near_frames = 0
        self.water_near_target = None
        self.water_near_observations = deque()
        self.water_recent_path = None
        self.water_recent_time = -1000000000.0
        self.spray_caution_s = 0.0
        self.near_curve_frames = 0
        self.near_curve_target = None
        self.near_curve_distance = 0.0
        self.near_curve_motion_s = 0.0
        self.near_curve_clear_s = 0.0

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

    def _bright_occlusion(self, raw_image, mask):
        """Measure bright coverage inside the previous visible road, not the sky.

        No location, effect identity or timer is used. The previous mask only
        defines this short-lived measurement region; it is never a new label.
        """
        if raw_image is None:
            return 0.0
        if raw_image.ndim != 3 or raw_image.shape[2] != 3 or raw_image.dtype != np.uint8:
            raise ValueError('Expected a uint8 camera image for occlusion checks')
        h, w = raw_image.shape[:2]
        reference = self.last_visible_mask if self.last_visible_mask is not None and self.elapsed_s - self.last_visible_mask_s <= 0.3 else mask
        region = cv2.resize((reference == 1).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        region[:int(0.5 * h)] = 0
        region[int(0.93 * h):] = 0
        region = cv2.erode(region, np.ones((3, 3), np.uint8)) != 0
        if np.count_nonzero(region) < 100:
            return 0.0
        blue, green, red = cv2.split(raw_image)
        low = cv2.min(cv2.min(blue, green), red)
        high = cv2.max(cv2.max(blue, green), red)
        bright = cv2.bitwise_and(cv2.inRange(low, 211, 255), cv2.inRange(cv2.subtract(high, low), 0, 34)) != 0
        return float(bright[region].mean())

    def _gap_envelope(self, runs, row_index, width, yaw_rate):
        """Track both outer edges when one connected road has internal holes.

        Require recent matching edges, modest gaps and visible road on both
        sides. Separate components are excluded before this check; large widening
        branches and edge shifts fail the matching-edge limits. The raw mask remains unchanged.
        """
        if len(runs) < 2 or self.last_geometry_centers is None or self.last_geometry_widths is None or (abs(yaw_rate) > 20):
            return None
        previous_center = self.last_geometry_centers[row_index]
        previous_width = self.last_geometry_widths[row_index]
        if previous_center is None or previous_width is None or previous_width < 0.08 * width:
            return None
        a, b = (runs[0][0], runs[-1][1])
        span = b - a + 1
        coverage = sum((right - left + 1 for left, right in runs)) / span
        largest_gap = max((right[0] - left[1] - 1 for left, right in zip(runs, runs[1:])))
        center = (a + b) / 2
        if coverage >= 0.55 and largest_gap <= 0.4 * span and (0.8 <= span / previous_width <= 1.2) and (abs(center - previous_center) <= 0.04 * width):
            return (a, b)
        return None

    def _visible_near_corridor(self, mask):
        """Current spatial evidence for low-speed partial-occlusion crawling."""
        h, w = mask.shape
        road = cv2.morphologyEx((mask == 1).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
        n, labels = cv2.connectedComponents(road)
        counts = np.bincount(labels[int(0.9 * h):, int(0.28 * w):int(0.72 * w)].ravel(), minlength=n)
        counts[0] = 0
        if counts.max() < 25:
            return None
        connected = labels == int(np.argmax(counts))
        runs = self._runs(self._road_slice(connected, int(0.85 * h)))
        if len(runs) != 1:
            return None
        a, b = runs[0]
        span = b - a + 1
        center = (a + b) / 2
        margin = min(w / 2 - a, b - w / 2) / span
        bottom = connected[int(0.88 * h):int(0.97 * h), int(0.43 * w):int(0.57 * w)]
        if a > 1 and b < w - 2 and (margin > 0.35) and (abs(center - w / 2) < 0.06 * w) and bottom.size and (bottom.mean() > 0.85):
            return float(center)
        return None

    def decide(self, mask, dt_s=0.083, actual_speed=0.0, yaw_rate=0.0, actual_angle=None, raw_image=None):
        self.current_cap_revalidation_s = max(0.0, self.current_cap_revalidation_s)
        self.curve_speed_control_info = dict(braking=False, reason='no_current_curve_speed_evaluation')
        decision = self._decide_current(mask, dt_s=dt_s, actual_speed=actual_speed, yaw_rate=yaw_rate, actual_angle=actual_angle, raw_image=raw_image)
        if 'cap_revalidated' not in self.curve_speed_control_info:
            self.current_cap_revalidation_s = 0.0
        ceiling = self.curve_speed_brake_ceiling
        if self.curve_speed_braking and ceiling is not None:
            if float(actual_speed) > float(ceiling) - 0.5:
                prior_mode = decision.mode
                decision = replace(decision, speed=0.0, mode='corner_momentum_brake')
                self.last_speed_cmd = 0.0
                self.curve_speed_control_info.update(braking=True, reason='measured_speed_obligation_retained', actual_speed=float(actual_speed), retained_brake_ceiling=float(ceiling), underlying_current_mode=prior_mode)
            else:
                self.curve_speed_braking = False
                self.curve_speed_brake_ceiling = None
                self.curve_speed_control_info.update(braking=False, reason='measured_speed_obligation_released', actual_speed=float(actual_speed), retained_brake_ceiling=None)
        return decision

    def _decide_current(self, mask, dt_s=0.083, actual_speed=0.0, yaw_rate=0.0, actual_angle=None, raw_image=None):
        self.brake_release_request.begin_observation()
        self.clipped_reference_info = dict(applied=False, reason='not_current_unambiguous_clipped_road')
        self.acceleration_profile_info = dict(rate=12.0, eligible=False, reason='not_current_resolved_straight')
        if mask.ndim != 2 or mask.dtype != np.uint8:
            raise ValueError('Road mask must be a 2D uint8 array')
        dt_s = float(np.clip(dt_s, 0.01, 0.25))
        self.elapsed_s += dt_s
        self.visibility_pulse_s = max(0.0, self.visibility_pulse_s - dt_s)
        self.visibility_cooldown_s = max(0.0, self.visibility_cooldown_s - dt_s)
        h, w = mask.shape
        occlusion_fraction = self._bright_occlusion(raw_image, mask)
        corridor = visible_spray_corridor(mask, raw_image)
        if corridor is not None:
            occlusion_fraction = max(occlusion_fraction, corridor['white_fraction'])
        self.spray_caution_s = max(0.0, self.spray_caution_s - dt_s)
        if occlusion_fraction >= 0.06:
            self.spray_caution_s = 1.0
        close_curve = current_near_curve_path(mask, raw_image)
        curve_hint = bool(close_curve and close_curve['minimum_angle'] != 0 and (occlusion_fraction < 0.12) and (abs(yaw_rate) < 35) and (self.near_curve_distance < 2.0) and (self.near_curve_motion_s < 3.0))
        if curve_hint:
            self.near_curve_frames = self.near_curve_frames + 1 if self.near_curve_target is not None and abs(close_curve['target'] - self.near_curve_target) < 0.08 * w else 1
            self.near_curve_target = close_curve['target']
            self.near_curve_clear_s = 0.0
        else:
            self.near_curve_frames = 0
            self.near_curve_target = None
            clear_center = close_curve is None and occlusion_fraction < 0.025 and (self.last_geometry_bounds is not None) and (min(w / 2 - self.last_geometry_bounds[0], self.last_geometry_bounds[1] - w / 2) / max(self.last_geometry_width, 1) > 0.35) and (abs(self.last_angle) < 6)
            self.near_curve_clear_s = self.near_curve_clear_s + dt_s if clear_center else 0
            if self.near_curve_clear_s > 0.5:
                self.near_curve_distance = 0
                self.near_curve_motion_s = 0
        if self.near_curve_frames >= 3:
            desired = close_curve['minimum_angle']
            angle = float(np.clip(desired, self.last_angle - 40 * dt_s, self.last_angle + 40 * dt_s))
            aligned = actual_angle is not None and abs(actual_angle - self.steer_sign * angle) < 5
            curve_request = 10.0 if close_curve['near_clearance'] > 0.28 else 6.0
            curve_ceiling = 13.0 if curve_request == 10 else 9.0
            request = curve_request if actual_speed <= curve_ceiling and aligned else 0.0
            if actual_speed > 0.5:
                self.near_curve_distance += actual_speed / 3.6 * dt_s
                self.near_curve_motion_s += dt_s
            self.last_angle = angle
            self.last_speed_cmd = request
            self.last_geometric_error = None
            self.error_rate = 0.0
            self.frames += 1
            return Decision(request, self.steer_sign * angle, close_curve['target'], 2 / 3, (close_curve['center'], close_curve['target'], None), (close_curve['width'], close_curve['target_width'], None), 0, *close_curve['near_bounds'], mode='near_curve_tracking', yaw_rate=float(yaw_rate), near_clearance=close_curve['near_clearance'], preview_quality=0.5, occlusion_fraction=occlusion_fraction, current_path=close_curve)
        partial_crawl = False
        if occlusion_fraction >= 0.12:
            self.occlusion_elapsed_s += dt_s
            self.occlusion_release_s = 0.16
            current_spray = corridor is not None and abs(yaw_rate) < 12 and (actual_angle is not None) and (abs(actual_angle) < 8)
            while self.spray_observations and self.elapsed_s - self.spray_observations[0][0] > 0.45:
                self.spray_observations.popleft()
            if abs(yaw_rate) >= 12 or actual_angle is None or abs(actual_angle) >= 8:
                self.spray_observations.clear()
            if current_spray:
                center = corridor['centers'][1]
                if self.spray_observations and abs(center - self.spray_observations[-1][1]) >= 0.025 * w:
                    self.spray_observations.clear()
                while self.spray_observations and abs(center - self.spray_observations[0][1]) >= 0.025 * w:
                    self.spray_observations.popleft()
                self.spray_observations.append((self.elapsed_s, center))
                self.spray_visible_frames = self.spray_visible_frames + 1 if self.spray_center is not None and abs(center - self.spray_center) < 0.025 * w else 1
                self.spray_center = center
            else:
                self.spray_visible_frames = 0
                self.spray_center = None
            if current_spray and (self.spray_visible_frames >= 3 or len(self.spray_observations) >= 3):
                near, mid, far = corridor['centers']
                desired = float(np.clip(14 * (mid - w / 2) / (w / 2) + 3 * (mid - near) / (w / 2) - 0.06 * yaw_rate, -6, 6))
                angle = float(np.clip(desired, self.last_angle - 10 * dt_s, self.last_angle + 10 * dt_s))
                flowing = corridor['clearance'] > 0.35 and abs(mid - w / 2) < 0.15 * w and (abs(far - mid) < 0.12 * w) and (abs(yaw_rate) < 8)
                flow_cap = band(18.0) if flowing else 6.0
                actual_cap = band(21.0) if flowing else 9.0
                speed = flow_cap if actual_speed <= actual_cap else 0.0
                self.last_angle = angle
                self.last_speed_cmd = speed
                self.last_target = mid
                self.last_geometric_error = None
                self.error_rate = 0.0
                self.last_geometry_bounds = corridor['near_bounds']
                self.last_geometry_center = near
                self.last_geometry_width = corridor['widths'][0]
                self.last_geometry_centers = corridor['centers']
                self.last_geometry_widths = corridor['widths']
                self.last_visible_mask = mask.copy()
                self.last_visible_mask_s = self.elapsed_s
                self.frames += 1
                return Decision(speed, self.steer_sign * angle, mid, 2 / 3, corridor['centers'], corridor['widths'], (far - mid) / (w / 2), *corridor['near_bounds'], mode='occlusion_track', yaw_rate=float(yaw_rate), near_clearance=corridor['clearance'], preview_quality=0.6, occlusion_fraction=occlusion_fraction, mask_gap_envelope=True)
            previous_clearance = 0.0 if self.last_geometry_bounds is None else min(w / 2 - self.last_geometry_bounds[0], self.last_geometry_bounds[1] - w / 2) / max(self.last_geometry_width, 1.0)
            if corridor is not None and self.occlusion_elapsed_s <= 0.22 and (previous_clearance > 0.35) and (corridor['clearance'] > 0.35) and (abs(corridor['centers'][1] - w / 2) < 0.15 * w) and (abs(corridor['centers'][2] - corridor['centers'][1]) < 0.12 * w) and (abs(yaw_rate) < 8) and (abs(self.last_angle) < 5) and (actual_angle is not None) and (abs(actual_angle) < 6) and (3 < actual_speed <= 13) and (not self.was_junction):
                self.frames += 1
                self.last_speed_cmd = min(10.0, float(actual_speed))
                return Decision(self.last_speed_cmd, self.steer_sign * self.last_angle, corridor['centers'][1], 2 / 3, corridor['centers'], corridor['widths'], 0, *corridor['near_bounds'], mode='occlusion_coast', yaw_rate=float(yaw_rate), near_clearance=corridor['clearance'], preview_quality=0.6, occlusion_fraction=occlusion_fraction, mask_gap_envelope=True)
            path = current_near_water_path(mask, raw_image)
            current_curve = close_curve
            if current_curve is not None and current_curve['minimum_angle'] != 0 and (path is None or (path['target'] - w / 2) * current_curve['minimum_angle'] > 0):
                path = current_curve
            elif path is None:
                path = current_curve
            held = False
            if path is None and self.elapsed_s - self.water_recent_time <= 0.25:
                now_path = current_near_water_path(mask, raw_image, allow_white_foot=True)
                if now_path is not None and self.water_recent_path is not None and (abs(now_path['target'] - self.water_recent_path['target']) < 0.08 * w):
                    path = self.water_recent_path
                    held = True
            near_stable = path is not None and abs(yaw_rate) < 25 and (self.water_recovery_s < 8.0) and (self.water_recovery_m < 4.0)
            obs = self.water_near_observations
            while obs and self.elapsed_s - obs[0][0] > 0.35:
                obs.popleft()
            if near_stable:
                if self.water_near_target is not None and abs(path['target'] - self.water_near_target) >= 0.08 * w:
                    obs.clear()
                obs.append((self.elapsed_s, float(path['target'])))
                self.water_near_frames = len(obs)
                self.water_near_target = path['target']
            elif not obs or abs(yaw_rate) >= 25 or self.water_recovery_s >= 8.0 or (self.water_recovery_m >= 4.0):
                obs.clear()
                self.water_near_frames = 0
                self.water_near_target = None
            if near_stable:
                if not held:
                    self.water_recent_path = path
                    self.water_recent_time = self.elapsed_s
                desired = float(np.clip(20 * (path['target'] - w / 2) / (w / 2) - 0.08 * yaw_rate, -12, 12))
                rate = 20.0
                if path.get('curve_recovery'):
                    desired = float(np.clip(35 * (path['target'] - w / 2) / (w / 2) - 0.08 * yaw_rate, -25, 25))
                    minimum = path['minimum_angle']
                    if minimum > 0:
                        desired = max(desired, minimum)
                    if minimum < 0:
                        desired = min(desired, minimum)
                    rate = 35.0
                angle = float(np.clip(desired, self.last_angle - rate * dt_s, self.last_angle + rate * dt_s))
                wheel_target = desired if path.get('curve_recovery') else angle
                aligned = actual_angle is not None and abs(float(actual_angle) - self.steer_sign * wheel_target) < 5
                near_request = 10.0 if path['near_clearance'] > 0.28 else 5.0 if not path.get('curve_recovery') else 4.0
                near_ceiling = 13.0 if near_request == 10 else 7.0
                strong_current = not held and path['near_clearance'] > 0.35 and (abs(yaw_rate) < 10) and (actual_speed < 5) and aligned
                confirmed = self.water_near_frames >= 3 or (strong_current and self.water_near_frames >= 2 and (self.elapsed_s - obs[0][0] >= 0.06))
                speed = near_request if confirmed and actual_speed < near_ceiling and aligned else 0.0
                if actual_speed > 0.5:
                    self.water_recovery_s += dt_s
                    self.water_recovery_m += max(0, float(actual_speed)) / 3.6 * dt_s
                self.last_angle = angle
                self.last_speed_cmd = speed
                self.last_target = path['target']
                self.last_geometric_error = None
                self.error_rate = 0.0
                self.frames += 1
                return Decision(speed, self.steer_sign * angle, path['target'], 2 / 3, (path['center'], path['target'], None), (path['width'], path['target_width'], None), 0, *path['near_bounds'], mode='occlusion_near_recovery', yaw_rate=float(yaw_rate), near_clearance=path['near_clearance'], occlusion_fraction=occlusion_fraction, current_path=dict(path, agreement_frames=self.water_near_frames, agreement_window_s=0.35, agreement_age_s=self.elapsed_s - obs[0][0]))
            center = self._visible_near_corridor(mask) if occlusion_fraction < 0.45 else None
            if center is not None and abs(yaw_rate) < 12:
                self.partial_visible_frames = self.partial_visible_frames + 1 if self.partial_center is not None and abs(center - self.partial_center) < 0.025 * w else 1
                self.partial_center = center
            else:
                self.partial_visible_frames = 0
                self.partial_center = None
            partial_crawl = self.partial_visible_frames >= 3 and actual_speed < 6 and (actual_angle is not None) and (abs(float(actual_angle) - self.steer_sign * self.last_angle) < 5)
            previous_clearance = 0.0 if self.last_geometry_bounds is None else min(w / 2 - self.last_geometry_bounds[0], self.last_geometry_bounds[1] - w / 2) / max(self.last_geometry_width, 1.0)
            if not partial_crawl and self.occlusion_elapsed_s <= 0.25 and (previous_clearance > 0.32) and (abs(yaw_rate) < 20) and (abs(self.last_angle) < 5) and (not self.was_junction):
                self.last_angle *= 0.8 ** (dt_s / 0.083)
                self.last_speed_cmd = 0.0
                self.error_rate = 0.0
                self.straight_s = 0.0
                self.frames += 1
                return Decision(0.0, self.steer_sign * self.last_angle, self.last_target, 0.5, self.last_geometry_centers, self.last_geometry_widths, 0.0, *self.last_geometry_bounds, mode='occlusion_hold', yaw_rate=float(yaw_rate), near_clearance=float(previous_clearance), occlusion_fraction=occlusion_fraction, occlusion_hold=True)
            if not partial_crawl:
                decision = self._lost(w, dt_s, reason='occlusion_stop')
                decision.occlusion_fraction = occlusion_fraction
                return decision
        else:
            self.occlusion_elapsed_s = 0.0
            obs = self.water_near_observations
            while obs and self.elapsed_s - obs[0][0] > 0.35:
                obs.popleft()
            self.water_near_frames = len(obs)
            if not obs:
                self.water_near_target = None
            self.spray_visible_frames = 0
            self.spray_center = None
            self.partial_visible_frames = 0
            self.partial_center = None
        self.occlusion_release_s = max(0.0, self.occlusion_release_s - dt_s)
        repaired, self.last_paint_holes = fill_enclosed_paint_holes(mask, raw_image)
        road = cv2.morphologyEx(repaired, cv2.MORPH_CLOSE, np.ones((3, 5), np.uint8))
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
        mask_gap_envelope = False
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
            envelope = self._gap_envelope(runs, len(centers), w, yaw_rate)
            if envelope is not None:
                a, b = envelope
                run_counts[-1] = 1
                mask_gap_envelope = True
            else:
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
        reference = current_clipped_reference(road, bounds[0]) if not junction and bounds[0] is not None and (not mask_gap_envelope) and (len(valid) == 3) and (occlusion_fraction < 0.025) and (abs(yaw_rate) < 35) else None
        if reference is not None:
            old = self.clipped_reference_previous
            same = old is not None and self.elapsed_s - old['time'] <= 0.15 and (abs(reference['center'] - old['center']) < 0.08 * w) and (abs(reference['estimated_width'] - old['width']) < 0.08 * w)
            self.clipped_reference_confirm_s = self.clipped_reference_confirm_s + dt_s if same else 0.0
            self.clipped_reference_previous = dict(time=self.elapsed_s, center=reference['center'], width=reference['estimated_width'])
            applied = self.clipped_reference_confirm_s >= 0.08
            self.clipped_reference_info = dict(reference, applied=applied, reason='same_current_width_steering_reference' if applied else 'confirm_current_width_shape', confirmation_s=self.clipped_reference_confirm_s, visible_midpoint=near)
            if applied:
                near = reference['center']
                center = 0.65 * near + 0.35 * middle
                near_slope = (middle - near) / (w / 2)
        else:
            self.clipped_reference_previous = None
            self.clipped_reference_confirm_s = 0.0
        near_bounds = bounds[0] if bounds[0] is not None else bounds[valid[0]]
        observed_bounds = near_bounds
        observed_width = near_width
        observed_clearance = min(w / 2 - observed_bounds[0], observed_bounds[1] - w / 2) / max(observed_width, 1.0)
        geometry_guard = False
        mask_shift_guard = False
        if self.last_geometry_center is not None and self.last_geometry_width:
            previous_clearance = min(w / 2 - self.last_geometry_bounds[0], self.last_geometry_bounds[1] - w / 2) / self.last_geometry_width
            width_ratio = min(near_width, self.last_geometry_width) / max(near_width, self.last_geometry_width)
            coherent_shift = False
            if len(valid) == 3 and self.last_geometry_centers is not None and all((v is not None for v in self.last_geometry_centers)) and (self.last_geometry_widths is not None) and all((v is not None for v in self.last_geometry_widths)) and (abs(yaw_rate) < 8.0) and all((a > 1 and b < w - 2 for a, b in bounds)):
                shifts = np.array(centers) - np.array(self.last_geometry_centers)
                width_ratios = np.minimum(widths, self.last_geometry_widths) / np.maximum(widths, self.last_geometry_widths)
                coherent_shift = np.ptp(shifts) < 0.025 * w and np.all(width_ratios > 0.9)
            shifted = abs(center - self.last_geometry_center) > 0.08 * w
            suspect = not junction and abs(yaw_rate) < 25.0 and shifted and (width_ratio < 0.8 or len(valid) < 3 or coherent_shift)
            if suspect:
                if self.pending_geometry_center is not None and abs(center - self.pending_geometry_center) < 0.04 * w:
                    self.pending_geometry_count += 1
                else:
                    self.pending_geometry_count = 1
                self.pending_geometry_center = center
                self.geometry_reject_s += dt_s
                geometry_guard = self.pending_geometry_count < 2 and self.geometry_reject_s <= 0.25 and (observed_clearance > 0.32) and (previous_clearance > 0.32)
                mask_shift_guard = bool(geometry_guard and coherent_shift)
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
            self.last_geometry_centers = tuple(centers)
            self.last_geometry_widths = tuple(widths)
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
        desired_angle = 18.2 * error + 3.0 * curvature + 2.0 * near_slope + derivative
        if hold_path:
            desired_angle -= float(np.clip(0.35 * self.junction_heading_error, -5.0, 5.0))
        safe_margin = max(0.1 * w, 0.26 * near_width)
        if left_margin < safe_margin:
            desired_angle += 10.0 * (safe_margin - left_margin) / safe_margin
        if right_margin < safe_margin:
            desired_angle -= 10.0 * (safe_margin - right_margin) / safe_margin
        steer_desired = desired_angle + 2.6 * error + 0.5 * curvature
        damping = 0.1 if junction else 0.06
        steer_desired -= float(np.clip(damping * yaw_rate, -5.0, 5.0))
        alpha = 1.0 - 0.35 ** (dt_s / 0.083)
        steering_limit = float(np.interp(float(actual_speed), (15.0, 30.0), (22.0, 16.0)))
        angle = float(np.clip((1.0 - alpha) * self.last_angle + alpha * steer_desired, -steering_limit, steering_limit))
        if mode_transition_guard:
            angle = float(np.clip(angle, self.last_angle - 18.0 * dt_s, self.last_angle + 18.0 * dt_s))
        turn = min(1.0, abs(desired_angle) / 13.5 + preview_bend * 1.8)
        profile_resolved = len(valid) == 3 and preview_quality >= 0.6 and (clearance > 0.3) and (observed_clearance > 0.3) and (occlusion_fraction < 0.06) and (not geometry_guard) and (not edge_guard) and (not mode_transition_guard)
        base_max = min(self.max_speed, band(40.0) if profile_resolved else 40.0)
        tracking_min = band(self.min_speed) if profile_resolved else self.min_speed
        speed = base_max - (base_max - tracking_min) * turn
        mode = 'normal'
        reversing_with_inertia = self.last_angle * steer_desired < 0.0 and abs(self.last_angle) > 5.0 and (abs(yaw_rate) > 35.0)
        hazard = target_jump or abs(yaw_rate) > 65.0 or reversing_with_inertia or (clearance < 0.18)
        if hazard:
            self.hazard_hold_s = 0.5
        else:
            self.hazard_hold_s = max(0.0, self.hazard_hold_s - dt_s)
        if preview_bend > 0.25:
            speed = min(speed, band(24.0))
            mode = 'corner'
        if preview_bend > 0.45:
            speed = min(speed, tracking_min)
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
            mild_hold = observed_clearance > 0.4 and previous_clearance > 0.4 and (abs(yaw_rate) < 8) and (abs(float(actual_angle or 0)) < 6) and (occlusion_fraction < 0.025)
            speed = min(speed, 12.0 if mild_hold else 8.0)
            mode = 'perception_guard'
        if mode_transition_guard:
            speed = min(speed, 10.0)
            mode = 'mode_transition'
        if edge_guard:
            speed = min(speed, 8.0)
            mode = 'edge_trend_guard'
        if mask_gap_envelope:
            speed = min(speed, 12.0)
            if mode == 'normal':
                mode = 'mask_gap_caution'
        if self.occlusion_release_s > 0:
            speed = min(speed, 8.0)
            if mode in ('normal', 'mask_gap_caution'):
                mode = 'occlusion_release'
        boundary_recovery_guard = observed_clearance < 0.28
        if boundary_recovery_guard:
            recovery_cap = float(np.interp(observed_clearance, (0.0, 0.18, 0.28), (3.0, 6.0, 12.0)))
            speed = min(speed, recovery_cap)
            if mode in ('normal', 'corner', 'recovery'):
                mode = 'boundary_recovery'
        steering_settle_guard = False
        if actual_angle is not None and np.isfinite(actual_angle):
            steering_error = abs(float(actual_angle) - self.steer_sign * angle)
            steering_settle_guard = bool(steering_error > 7.0)
            if steering_settle_guard:
                speed = min(speed, 4.0)
                if mode in ('normal', 'corner', 'recovery', 'boundary_recovery'):
                    mode = 'steering_settle'
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
        straight_resolved = mode == 'normal' and confidence == 1.0 and (preview_quality >= 0.9) and (preview_bend < 0.1) and (abs(near_slope) < 0.08) and (clearance > 0.36) and (abs(error) < 0.08) and (abs(yaw_rate) < 6.0) and (abs(angle) < 2.5) and (self.hazard_hold_s <= 0.0) and (not junction) and (not geometry_guard) and (not edge_guard) and (not mode_transition_guard) and (not clipped_near_preview)
        self.straight_s = self.straight_s + dt_s if straight_resolved else 0.0
        if straight_resolved:
            extra = max(0.0, self.max_speed - base_max)
            speed += extra * float(np.clip((self.straight_s - 0.35) / 0.4, 0.0, 1.0))
        if self.elapsed_s < 3.32:
            speed = min(speed, self.min_speed + self.elapsed_s * 3.0)
        momentum_risk = preview_bend > 0.25 or abs(near_slope) > 0.25 or junction or clipped_near_preview or geometry_guard or edge_guard or mode_transition_guard or (confidence < 1) or (self.hazard_hold_s > 0) or boundary_recovery_guard or steering_settle_guard
        momentum_risk = momentum_risk or mask_gap_envelope or self.occlusion_release_s > 0
        tolerance = 2.0 if momentum_risk else 5.0
        curve_observed = preview_bend > 0.25 and confidence >= 2 / 3 and (preview_quality >= 0.85 or (self.curve_speed_braking and preview_quality >= 0.25)) and (not junction) and (not clipped_near_preview)
        current_normal_road = mode == 'normal' and confidence == 1 and (preview_quality >= 0.85) and (observed_clearance > 0.35) and (occlusion_fraction < 0.025) and (not junction) and (not geometry_guard) and (not clipped_near_preview)
        resolved_offcentre = current_normal_road and abs(error) > 0.08
        curve_observed = curve_observed or resolved_offcentre
        curve_ceiling = float(speed)
        prior_brake_ceiling = self.curve_speed_brake_ceiling
        cap_reobserved = bool(current_normal_road and (not edge_guard) and (not boundary_recovery_guard) and (not steering_settle_guard) and (not mode_transition_guard) and (not mask_gap_envelope) and (actual_angle is not None) and np.isfinite(actual_angle) and (abs(float(actual_angle) - self.steer_sign * angle) < 3) and (abs(yaw_rate) < 8) and (actual_speed < curve_ceiling - 0.5))
        self.current_cap_revalidation_s = self.current_cap_revalidation_s + dt_s if cap_reobserved else 0.0
        cap_revalidated = bool(self.current_cap_revalidation_s >= 0.12)
        if cap_revalidated:
            prior_brake_ceiling = None
            self.curve_speed_braking = False
        brake_continuation = bool(self.curve_speed_braking and prior_brake_ceiling is not None and (actual_speed > float(prior_brake_ceiling) - 0.5))
        if brake_continuation:
            retained_ceiling = float(prior_brake_ceiling)
            if curve_observed and curve_ceiling > 0:
                retained_ceiling = min(retained_ceiling, curve_ceiling)
        else:
            retained_ceiling = curve_ceiling
        brake_entry = bool(curve_observed and curve_ceiling > 0 and (actual_speed > curve_ceiling + 2.0))
        curve_brake = brake_entry or brake_continuation
        self.curve_speed_braking = curve_brake
        self.curve_speed_brake_ceiling = retained_ceiling if curve_brake else None
        self.curve_speed_control_info = dict(current_curve_observed=bool(curve_observed), planned_request_ceiling=curve_ceiling, actual_speed=float(actual_speed), braking=curve_brake, enter_excess_kmh=2.0, exit_reserve_kmh=0.5, reason='measured_current_path_momentum_brake' if curve_brake else 'current_curve_speed_ready', offcentre_current_road=bool(resolved_offcentre), current_normal_road=bool(current_normal_road), brake_continuation=bool(brake_continuation), retained_brake_ceiling=self.curve_speed_brake_ceiling, cap_revalidated=cap_revalidated, current_cap_revalidation_s=self.current_cap_revalidation_s)
        overspeed = max(0.0, float(actual_speed) - speed - tolerance)
        overspeed_caution = bool(momentum_risk and overspeed > 0)
        speed = max(0.0, speed - (0.8 if momentum_risk else 0.4) * overspeed)
        if curve_brake:
            speed = 0.0
            mode = 'corner_momentum_brake'
        fast_accel = bool(straight_resolved and self.straight_s >= 0.2 and (occlusion_fraction < 0.025) and (self.spray_caution_s <= 0) and (actual_angle is not None) and np.isfinite(actual_angle) and (abs(float(actual_angle) - self.steer_sign * angle) < 3))
        accel_rate = 24.0 if fast_accel else 12.0
        self.acceleration_profile_info = dict(rate=accel_rate, eligible=fast_accel, previous_request=float(self.last_speed_cmd), before_slew=float(speed), actual_speed=float(actual_speed), straight_evidence_s=float(self.straight_s))
        physical_ceiling = speed
        speed = min(speed, self.last_speed_cmd + accel_rate * dt_s)
        release_eligible = mode == 'normal' and confidence == 1 and (preview_quality >= 0.85) and (preview_bend < 0.18) and (abs(near_slope) < 0.15) and (abs(error) < 0.12) and (clearance > 0.35) and (observed_clearance > 0.35) and (occlusion_fraction < 0.025) and (self.spray_caution_s <= 0) and (not junction) and (not clipped_near_preview) and (not geometry_guard) and (not edge_guard) and (not mode_transition_guard) and (not mask_gap_envelope) and (not boundary_recovery_guard) and (not steering_settle_guard) and (not overspeed_caution) and (self.hazard_hold_s <= 0) and (self.occlusion_release_s <= 0) and (self.visibility_pulse_s <= 0)
        curve_release = bool(curve_observed and (not curve_brake) and (mode == 'corner') and (observed_clearance > 0.35) and (not geometry_guard) and (not edge_guard) and (not boundary_recovery_guard) and (not steering_settle_guard) and (occlusion_fraction < 0.025) and (not mask_gap_envelope))
        speed = self.brake_release_request.apply(speed, physical_ceiling, dt=dt_s, actual_speed=float(actual_speed), wheel=actual_angle, angle=self.steer_sign * angle, yaw=yaw_rate, eligible=release_eligible or curve_release, curve_release=curve_release)
        spray_prealert = self.spray_caution_s > 0.0
        if spray_prealert:
            flowing = confidence == 1 and clearance > 0.35 and (preview_quality >= 0.6) and (abs(angle) < 6) and (abs(yaw_rate) < 8) and (not junction) and (not geometry_guard) and (not edge_guard) and (not steering_settle_guard)
            spray_cap = band(16.0) if flowing else 8.0
            speed = 0.0 if actual_speed > spray_cap + 3 else min(speed, spray_cap)
            if mode in ('normal', 'corner', 'mask_gap_caution', 'occlusion_release'):
                mode = 'spray_prealert'
        if partial_crawl:
            angle = float(np.clip(16 * (self.partial_center - w / 2) / (w / 2) - 0.06 * yaw_rate, -4, 4))
            speed = min(speed, 2.0)
            mode = 'occlusion_crawl'
        clear_observed = occlusion_fraction < 0.12 and confidence == 1 and (preview_quality >= 0.9) and (clearance > 0.3) and (not junction)
        self.water_clear_s = self.water_clear_s + dt_s if clear_observed else 0.0
        if self.water_clear_s > 0.5:
            self.water_recovery_s = 0.0
            self.water_recovery_m = 0.0
            self.water_near_frames = 0
            self.water_near_target = None
            self.water_near_observations.clear()
        self.last_geometric_error = geometric_error
        self.last_angle = angle
        self.last_target = target
        self.last_speed_cmd = speed
        self.frames += 1
        if raw_image is not None:
            self.last_visible_mask = mask.copy()
            self.last_visible_mask_s = self.elapsed_s
        return Decision(float(speed), float(self.steer_sign * angle), target, confidence, tuple(centers), tuple(widths), float(curvature), *near_bounds, mode, float(preview_bend), bool(junction), float(yaw_rate), float(clearance), float(self.bias_applied), float(preview_quality), bool(hold_path), float(self.junction_heading_error), bool(geometry_guard), bool(edge_guard), bool(forward_path), bool(mode_transition_guard), bool(geometry_target_hold), bool(clipped_near_preview), float(derivative), bool(overspeed_caution), mask_shift_guard, boundary_recovery_guard, steering_settle_guard, occlusion_fraction, False, mask_gap_envelope, spray_prealert)

    def _lost(self, w, dt_s=0.083, reason='mask_lost'):
        self.brake_release_request.clear_s = 0.0
        self.brake_release_request.evaluated = False
        self.brake_release_request.info = dict(applied=False, reason=reason)
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
        self.last_geometry_centers = None
        self.last_geometry_widths = None
        self.pending_geometry_center = None
        self.pending_geometry_count = 0
        self.geometry_reject_s = 0.0
        self.last_margins = None
        self.margin_rates[:] = 0
        self.last_resolved_curvature = 0.0
        self.preview_missing_s = 0.0
        self.last_path_mode = None
        self.transition_remaining_s = 0.0
        self.last_visible_mask = None
        return Decision(0.0, float(self.steer_sign * self.last_angle), self.last_target if self.last_target is not None else w / 2, 0.0, (None, None, None), (None, None, None), 0.0, None, None, reason)
