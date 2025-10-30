from lightning.pytorch.callbacks import WeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

class EMAWeightAveraging(WeightAveraging):
    def __init__(
        self,
        decay: float = 0.999,
        start_step: int = 100,      # your default: start after 100 steps
        every_n_steps: int = 1,     # update cadence
        use_buffers: bool = True,   # also average buffers like BN stats
    ):
        super().__init__(avg_fn=get_ema_avg_fn(decay=decay), use_buffers=use_buffers)
        self.start_step = int(start_step)
        self.every_n_steps = int(every_n_steps)

    def should_update(self, step_idx=None, epoch_idx=None) -> bool:
        # Update on steps only; begin after start_step; then every_n_steps
        return (
            step_idx is not None
            and step_idx >= self.start_step
            and (step_idx - self.start_step) % self.every_n_steps == 0
        )
