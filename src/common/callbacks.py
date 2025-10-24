import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

class ProgressPrinter(BaseCallback):
    """Print a short progress line every print_freq steps."""
    def __init__(self, print_freq: int = 50000, window: int = 100):
        super().__init__()
        self.print_freq = int(print_freq)
        self.window = int(window)
        self._episode_returns = deque(maxlen=self.window)
        self._episode_lengths = deque(maxlen=self.window)
        self._last_print_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print_step >= self.print_freq:
            log = self.model.logger.name_to_value
            er = log.get("rollout/ep_rew_mean")
            el = log.get("rollout/ep_len_mean")
            msg = f"[{self.num_timesteps}]"
            if er is not None: msg += f" ep_rew_mean={er:.2f}"
            if el is not None: msg += f" ep_len_mean={int(el)}"
            print(msg)
            self._last_print_step = self.num_timesteps
        return True

class LatestModelSaver(BaseCallback):
    """Overwrite *_latest.zip every save_freq steps."""
    def __init__(self, save_path: str, save_freq: int = 100000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = int(save_freq)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            if self.verbose: print(f"[save] latest -> {self.save_path}")
            self.model.save(self.save_path)
        return True

class BestModelSaver(BaseCallback):
    """Save *_best.zip when ep_rew_mean improves."""
    def __init__(self, save_path: str, min_improve: float = 1e-6, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.min_improve = float(min_improve)
        self.best = None
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def _on_step(self) -> bool:
        cur = self.model.logger.name_to_value.get("rollout/ep_rew_mean")
        if cur is None: return True
        if self.best is None or cur > self.best + self.min_improve:
            self.best = cur
            if self.verbose: print(f"[save] best ({self.best:.2f}) -> {self.save_path}")
            self.model.save(self.save_path)
        return True

class PeriodicCheckpointSaver(BaseCallback):
    """Numbered checkpoints every save_freq into models/.../checkpoints/."""
    def __init__(self, folder: str, save_freq: int = 500000, prefix: str = "ckpt", verbose: int = 0):
        super().__init__(verbose)
        self.folder = folder
        self.save_freq = int(save_freq)
        self.prefix = prefix
        os.makedirs(self.folder, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.folder, f"{self.prefix}_{self.num_timesteps}.zip")
            if self.verbose: print(f"[save] checkpoint -> {path}")
            self.model.save(path)
        return True

