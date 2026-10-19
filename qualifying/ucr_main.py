"""UCR 2026 bot: real camera -> road model -> steering and speed."""
from ucr_lib import GetStatus, GetRaw, AVControl, CloseSocket
import os
from pathlib import Path
import signal
import time
import numpy as np
from predict import RoadPredictor
from road_control import RoadController

def main():

    def stop_signal(*unused):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    base = Path(__file__).resolve().parent
    predictor = RoadPredictor(base / 'road_selected.pth', color_order='BGR')
    predictor.predict(np.zeros((predictor.size[1], predictor.size[0], 3), dtype=np.uint8))
    controller = RoadController(steer_sign=float(os.environ.get('UCR_STEER_SIGN', '1')), right_bias=float(os.environ.get('UCR_RIGHT_BIAS', '0')), min_speed=float(os.environ.get('UCR_MIN_SPEED', '12')), max_speed=float(os.environ.get('UCR_MAX_SPEED', '25')))
    frame_id = 0
    status_every = 1
    state = None
    status_received_at = 0.0
    last_time = time.monotonic()
    previous_heading = None
    heading_time = None
    yaw_rate = 0.0
    print('SẴN SÀNG', flush=True)
    try:
        while True:
            loop_start = time.monotonic()
            status_fetched = state is None or frame_id % status_every == 0 or loop_start - status_received_at > 0.2
            if status_fetched:
                state = GetStatus()
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
            raw_image = GetRaw()
            got_raw = time.monotonic()
            mask = predictor.predict(raw_image)
            got_mask = time.monotonic()
            decision = controller.decide(mask, dt_s=got_mask - last_time, actual_speed=state.get('Speed', 0.0), yaw_rate=yaw_rate)
            got_decision = time.monotonic()
            AVControl(decision.speed, decision.angle)
            now = time.monotonic()
            last_time = now
            frame_id += 1
    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise
    finally:
        try:
            AVControl(0.0, 0.0)
        except Exception:
            pass
        CloseSocket()
if __name__ == '__main__':
    main()
