"""Every control cycle dispatches the queued command via fresh GetStatus."""
import math


def may_reuse_status(decision, age_s, yaw_rate, running_s, reused_last, speed):
    # GetStatus is also the ONLY verified transport for the queued AV command.
    # Reusing the response would drop a changed intermediate command.
    return False


def telemetry_speed_for_control(speed, age_s, cached):
    # This is an explicit caution buffer, not a measured acceleration bound.
    # It prevents reused speed from permitting extra cruise acceleration.
    return float(speed) + min(1.5, 15.0 * max(0.0, age_s)) if cached else float(speed)
