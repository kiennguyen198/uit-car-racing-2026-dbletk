from competition_console import ready
'UCR 2026 bot: real camera -> road model -> steering and speed.'
from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import os
from pathlib import Path
import signal
import time
import numpy as np
from predict import RoadPredictor
from detector import SignObstacleDetector
from traffic_control import TrafficController
from dataclasses import asdict, replace
from road_control import RoadController
from status_policy import may_reuse_status, telemetry_speed_for_control
from command_dispatch import CommandDispatch
from stuck_recovery import StuckRecovery
from motion_policy import VisibleMotionPolicy
from fountain_policy import FountainPolicy
from center_handover_history import adopt_center_handover_history
from process_ack import ProcessAck

def main():

    def stop_signal(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    base = Path(__file__).resolve().parent
    predictor = RoadPredictor(base / 'road_selected.pth', color_order='BGR')
    predictor.predict(np.zeros((predictor.size[1], predictor.size[0], 3), dtype=np.uint8))
    detector = SignObstacleDetector(base / 'map2_signs_obstacles_fountain_yolo11s_v9_clean.torchscript', base / 'map2_signs_obstacles_fountain_yolo11s_v9_clean.json')
    detector.warmup(5)
    controller_options = dict(steer_sign=float(os.environ.get('UCR_STEER_SIGN', '1')), right_bias=float(os.environ.get('UCR_RIGHT_BIAS', '0')), min_speed=float(os.environ.get('UCR_MIN_SPEED', '18')), max_speed=float(os.environ.get('UCR_MAX_SPEED', '50')))
    controller = RoadController(**controller_options)
    motion_policy = VisibleMotionPolicy()
    fountain_policy = FountainPolicy()
    traffic = TrafficController()
    recovery = StuckRecovery(allow_reverse=os.environ.get('UCR_ALLOW_SHORT_REVERSE', '0') == '1')
    frame_id = 0
    previous_decision = None
    dispatch = CommandDispatch()
    ack_worker = ProcessAck()
    reused_last = False
    state = None
    status_received_at = 0.0
    last_time = time.monotonic()
    previous_heading = None
    heading_time = None
    yaw_rate = 0.0
    ready()
    try:
        while True:
            loop_start = time.monotonic()
            status_fetched = not may_reuse_status(previous_decision, loop_start - status_received_at, yaw_rate, controller.elapsed_s, reused_last, float(state.get('Speed', 0.0)) if state is not None else float('nan'))
            if status_fetched:
                dispatch_before = time.monotonic()
                ack_worker.begin()
                state = GetStatus()
                dispatch.sent(dispatch_before, time.monotonic())
                status_received_at = time.monotonic()
                heading = state.get('Heading')
                if heading is not None:
                    if previous_heading is not None:
                        heading_dt = status_received_at - heading_time
                        delta = (float(heading) - previous_heading + 180.0) % 360.0 - 180.0
                        measured = float(np.clip(delta / max(heading_dt, 0.01), -150, 150))
                        yaw_rate += heading_dt / (0.08 + heading_dt) * (measured - yaw_rate)
                    previous_heading, heading_time = (float(heading), status_received_at)
            got_status = time.monotonic()
            reused_last = not status_fetched
            ack_worker.begin()
            raw_image = GetRaw()
            got_raw = time.monotonic()
            ack_worker.pause()
            mask = predictor.predict(raw_image)
            got_road_mask = time.monotonic()
            detector_start = time.monotonic()
            detections = detector.predict(raw_image)
            got_mask = time.monotonic()
            status_age = got_mask - status_received_at
            road_start = time.monotonic()
            control_speed = telemetry_speed_for_control(state.get('Speed', 0.0), status_age, reused_last)
            if status_age > 0.12:
                decision = controller._lost(mask.shape[1], got_mask - last_time, reason='status_stale')
            else:
                decision = controller.decide(mask, dt_s=got_mask - last_time, actual_speed=control_speed, yaw_rate=yaw_rate, actual_angle=state.get('Angle'), raw_image=raw_image)
            road_end = time.monotonic()
            decision = fountain_policy.apply(decision, mask, raw_image, detections, state, dt_s=got_mask - last_time, fresh=status_fetched and status_age <= 0.12, yaw_rate=yaw_rate, traffic_context=traffic.info)
            if fountain_policy.info.get('owns_path'):
                controller.last_angle = decision.angle / controller.steer_sign
                controller.last_speed_cmd = decision.speed
                controller.last_target = decision.target_x
                controller.last_geometric_error = None
                controller.error_rate = 0.0
            base_decision = decision
            decision = traffic.update(detections, mask, decision, raw_shape=raw_image.shape, dt_s=got_mask - last_time, speed=control_speed, yaw_rate=yaw_rate, actual_angle=state.get('Angle'), raw_image=raw_image)
            traffic_end = time.monotonic()
            decision = recovery.update(mask, raw_image, state, decision, traffic.info, dt_s=got_mask - last_time, fresh=status_fetched and status_age <= 0.12)
            decision = motion_policy.apply(decision, base_decision, traffic.info, mask, raw_image, state, dt_s=got_mask - last_time, fresh=status_fetched and status_age <= 0.12)
            recovery_end = time.monotonic()
            final_age = recovery_end - status_received_at
            if final_age > 0.12:
                decision = replace(decision, speed=0.0, mode='final_status_stale')
            previous_decision = decision
            got_decision = time.monotonic()
            AVControl(decision.speed, decision.angle)
            dispatch.queue(frame_id, decision.speed, decision.angle, time.monotonic())
            adopt_center_handover_history(controller, decision, base_decision, traffic.info, fresh=status_fetched and final_age <= 0.12)
            now = time.monotonic()
            last_time = now
            frame_id += 1
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise
    finally:
        ack_worker.close()
        try:
            AVControl(0.0, 0.0)
        except Exception:
            pass
        CloseSocket()
if __name__ == '__main__':
    main()
