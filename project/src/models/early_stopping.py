"""Strict best checkpoint tracking with a separate patience reference."""
from dataclasses import dataclass, asdict
import math

@dataclass
class EarlyStopping:
    patience: int = 5
    min_epochs: int = 5
    min_delta_abs: float = 1e-5
    min_delta_rel: float = .001
    enabled: bool = True
    best_metric: float = math.inf
    best_epoch: int = 0
    patience_reference: float = math.inf
    bad_epochs: int = 0
    stopped: bool = False

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise FloatingPointError(f'Nonfinite validation loss at epoch {epoch}: {value}')
        best = value < self.best_metric
        if best:
            self.best_metric, self.best_epoch = float(value), epoch
        reference = (not math.isfinite(self.patience_reference) or
                     self.patience_reference - value > self.min_delta_abs + self.min_delta_rel * abs(self.patience_reference))
        if reference:
            self.patience_reference, self.bad_epochs = float(value), 0
        else:
            self.bad_epochs += 1
        self.stopped = self.enabled and epoch >= self.min_epochs and self.bad_epochs >= self.patience
        return best, reference

    def state_dict(self):
        return asdict(self)
