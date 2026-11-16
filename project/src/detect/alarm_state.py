"""Causal alarm state machine shared by benchmarks and the deployed monitor."""
from dataclasses import dataclass

@dataclass
class AlarmMachine:
    threshold: float
    off_ratio: float = .9
    on_count: int = 2
    recovery_count: int = 5
    merge_gap_s: float = 3.
    consolidate: bool = True
    state: str = 'WARMUP'
    alarm_id: str | None = None
    pending: int = 0
    recovery: int = 0
    last_positive: float | None = None
    recovering_since: float | None = None
    first_trigger_at: str | None = None

    def update(self, score, sample_id, timestamp, seconds, warmup=False, quality=False):
        opened = closed = False
        if quality or warmup:
            self.state = 'DATA_QUALITY_HOLD' if quality else 'WARMUP'
            self.pending = self.recovery = 0
            self.alarm_id = None
            return dict(state=self.state, alarm_id=None, opened=False, closed=False, first_trigger_at=None)
        positive = score > self.threshold
        if not self.consolidate:
            self.state = 'ACTIVE' if positive else 'NORMAL'
            self.alarm_id = 'alarm:' + sample_id if positive else None
            return dict(state=self.state, alarm_id=self.alarm_id, opened=positive, closed=False, first_trigger_at=timestamp if positive else None)
        if self.state in ('WARMUP', 'DATA_QUALITY_HOLD'):
            self.state = 'NORMAL'
        if self.state in ('NORMAL', 'PENDING'):
            self.pending = self.pending + 1 if positive else 0
            self.state = 'PENDING' if self.pending else 'NORMAL'
            if self.pending >= self.on_count:
                self.state, self.alarm_id = 'ACTIVE', 'alarm:' + sample_id
                self.first_trigger_at, self.last_positive = timestamp, seconds
                self.recovery, self.recovering_since, opened = 0, None, True
        elif self.state in ('ACTIVE', 'RECOVERING'):
            if positive:
                self.last_positive, self.recovery = seconds, 0
                self.recovering_since, self.state = None, 'ACTIVE'
            elif score <= self.threshold - abs(self.threshold)*(1-self.off_ratio):
                self.recovery += 1
                if self.recovery >= self.recovery_count:
                    if self.recovering_since is None:
                        self.recovering_since = seconds
                    self.state = 'RECOVERING'
                    if seconds-self.recovering_since >= self.merge_gap_s:
                        self.state, closed = 'NORMAL', True
            else:
                self.recovery, self.recovering_since, self.state = 0, None, 'ACTIVE'
        result = dict(state=self.state, alarm_id=self.alarm_id, opened=opened, closed=closed,
                      first_trigger_at=self.first_trigger_at)
        if closed:
            self.alarm_id, self.first_trigger_at, self.pending = None, None, 0
        return result


def machine(threshold, policy, consolidate=True):
    return AlarmMachine(threshold, policy['off_ratio'], policy['on_count'], policy['recovery_count'], policy['merge_gap_s'], consolidate)
