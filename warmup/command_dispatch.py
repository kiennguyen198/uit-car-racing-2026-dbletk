"""Track queue and verified sequential Status dispatch, without claiming actuator ACK."""
class CommandDispatch:
    def __init__(self):
        self.pending = None
        self.last_sent = None
        self.last_sent_seq = None
        self.last_queue_age_ms = None
        self.status_roundtrip_ms = None
        self.seq = 0
        self.overwrites = 0

    def queue(self, frame, speed, angle, now):
        if self.pending is not None:
            self.overwrites += 1
        self.pending = dict(seq=self.seq, frame=frame, speed=float(speed),
                            angle=float(angle), queued_at=now)
        self.seq += 1

    def sent(self, request_at, returned_at):
        # Source audit: Status request serializes AVControl globals. Socket
        # return is not a physical wheel ACK or a camera exposure timestamp.
        self.last_sent = self.pending
        self.last_sent_seq = self.pending['seq'] if self.pending else None
        self.last_queue_age_ms = ((request_at-self.pending['queued_at'])*1000
                                  if self.pending else None)
        self.status_roundtrip_ms = (returned_at-request_at)*1000
        self.pending = None

    def trace(self):
        return dict(outbound_command=self.last_sent,
                    queue_age_ms_at_request=self.last_queue_age_ms,
                    status_roundtrip_ms=self.status_roundtrip_ms,
                    overwritten_commands=self.overwrites,
                    physical_ack_available=False, exposure_timestamp_available=False)
