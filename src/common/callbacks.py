from __future__ import annotations
from typing import Deque, List, Optional, Tuple, Dict
from collections import deque, defaultdict
import os
import csv
import time

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


class ProgressPrinter(BaseCallback):
    """
    Print one compact line every `print_freq` steps.
    Uses VecMonitor infos to compute rolling means.
    """
    def __init__(self, print_freq: int = 50_000, window: int = 100):
        super().__init__()
        self.print_freq = int(print_freq)
        self.window = int(window)
        self._episode_returns: Deque[float] = deque(maxlen=self.window)
        self._episode_lengths: Deque[int] = deque(maxlen=self.window)
        self._last_print_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos or []:
            ep = info.get("episode")
            if ep:
                self._episode_returns.append(float(ep.get("r", 0.0)))
                self._episode_lengths.append(int(ep.get("l", 0)))
        if (self.num_timesteps - self._last_print_step) >= self.print_freq:
            self._last_print_step = self.num_timesteps
            rew = safe_mean(list(self._episode_returns)) if self._episode_returns else float("nan")
            ln  = safe_mean(list(self._episode_lengths)) if self._episode_lengths else float("nan")
            print(f"[train] steps={self.num_timesteps:,} ep_rew_mean={rew:.2f} ep_len_mean={ln:.0f}")
        return True


class LatestModelSaver(BaseCallback):
    """Always save a 'latest' model to `save_dir`."""
    def __init__(self, save_dir: str, filename: str | None = None):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.filename = filename or "latest"

    def _on_step(self) -> bool:
        path = os.path.join(self.save_dir, self.filename)
        self.model.save(path)
        return True


class BestModelSaver(BaseCallback):
    """Track best mean reward over a small window; save as 'best'."""
    def __init__(self, save_dir: str, window: int = 100, filename: str | None = None):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.window = int(window)
        self.filename = filename or "best"
        self._ep_returns: Deque[float] = deque(maxlen=self.window)
        self._best = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos or []:
            ep = info.get("episode")
            if ep:
                self._ep_returns.append(float(ep.get("r", 0.0)))
        if len(self._ep_returns) >= max(5, self.window // 2):
            mean_r = safe_mean(list(self._ep_returns))
            if self._best is None or mean_r > self._best:
                self._best = mean_r
                path = os.path.join(self.save_dir, self.filename)
                self.model.save(path)
        return True


class PeriodicCheckpointSaver(BaseCallback):
    """Save checkpoints every `save_freq` env-steps."""
    def __init__(self, save_dir: str, save_freq: int = 1_000_000, prefix: str = ""):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_freq = int(save_freq)
        self.prefix = prefix or self.__class__.__name__

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"{self.prefix}_{self.num_timesteps}")
            self.model.save(path)
        return True


class EpisodeMetricsLogger(BaseCallback):
    """
    NEW: Collect per-episode custom metrics DURING TRAINING.

    - Reads custom counters from infos[i]['metrics'] (e.g., apples_eaten, pellets_eaten, dots_eaten, etc.)
    - Accumulates them over the episode.
    - When VecMonitor ends an episode (infos[i]['episode'] present), writes a row to <log_dir>/episodes_train.csv:
        app, algo, persona, episode, ep_len, ep_return, <custom metric columns...>

    This is CSV-only; it does not touch your SB3 logger or training loop semantics.
    """
    def __init__(self, log_dir: str, app: str, algo: str, persona: str):
        super().__init__()
        self.log_dir = str(log_dir)
        self.app = app
        self.algo = algo
        self.persona = persona

        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(self.log_dir, "episodes_train.csv")
        self._writer: csv.DictWriter | None = None
        self._accum: List[Dict[str, float]] = []

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._accum = [defaultdict(float) for _ in range(n_envs)]
        return True

    def _ensure_writer(self, row: Dict[str, float]) -> None:
        if self._writer is None:
            f = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            self._writer.writeheader()
            return
        # If new keys appeared mid-run, expand header safely
        missing = [k for k in row.keys() if k not in self._writer.fieldnames]
        if missing:
            # Read old, rewrite with expanded header
            with open(self._csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            fieldnames = self._writer.fieldnames + missing
            with open(self._csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            # Reopen append with new schema
            f = open(self._csv_path, "a", newline="")
            self._writer = csv.DictWriter(f, fieldnames=fieldnames)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for i, info in enumerate(infos):
            # accumulate custom metrics during the episode
            m = info.get("metrics")
            if m:
                acc = self._accum[i]
                for k, v in m.items():
                    try:
                        acc[k] += float(v)
                    except Exception:
                        pass

            # episode end: VecMonitor injects info["episode"]={"r":..,"l":..}
            ep = info.get("episode")
            if ep:
                row = {
                    "app": self.app,
                    "algo": self.algo,
                    "persona": self.persona,
                    "episode": int(info.get("episode_number", 0)),
                    "ep_len": int(ep.get("l", 0)),
                    "ep_return": float(ep.get("r", 0.0)),
                    "timestamp_ms": int(time.time() * 1000),
                }
                # add accumulated custom metrics for this env
                for k, v in self._accum[i].items():
                    row[k] = v
                # reset env accumulator
                self._accum[i].clear()

                self._ensure_writer(row)
                self._writer.writerow(row)

        return True
