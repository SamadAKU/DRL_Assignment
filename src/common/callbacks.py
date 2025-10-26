from __future__ import annotations
from typing import Deque, List, Dict
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
        for info in infos:
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
        for info in infos:
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
    Collect per-episode custom metrics during training and write to CSV.

    - Reads counters from infos[i]['metrics'] (e.g., pellets, deaths, unique_tiles, score_delta, truncations, episode_ale_score).
    - Accumulates over the episode.
    - On episode end, writes a row to <log_dir>/episodes_train.csv with a stable, expanding header.
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
        self._fh = None
        self._accum: List[Dict[str, float]] = []

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._accum = [defaultdict(float) for _ in range(n_envs)]
        return True

    def _ensure_writer(self, row: Dict[str, float]) -> None:
        # initial fixed columns + any discovered metric names
        base_cols = [
            "app", "algo", "persona", "episode",
            "ep_len", "ep_return", "timestamp_ms",
        ]
        metric_cols = sorted([k for k in row.keys() if k not in base_cols])
        cols = base_cols + metric_cols

        if self._writer is None:
            self._fh = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=cols)
            self._writer.writeheader()
            return

        # Expand header deterministically if new columns appear
        current = list(self._writer.fieldnames)
        if set(cols) - set(current):
            all_cols = current + [c for c in cols if c not in current]
            # read old data
            self._fh.close()
            with open(self._csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            # rewrite with new header
            with open(self._csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=all_cols)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            # reopen for append
            self._fh = open(self._csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=all_cols)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for i, info in enumerate(infos):
            m = info.get("metrics")
            if m:
                acc = self._accum[i]
                for k, v in m.items():
                    # convert to float if possible; else keep original
                    acc[k] = float(v) if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace('.','',1).isdigit()) else v

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
                for k, v in self._accum[i].items():
                    row[k] = v
                self._accum[i].clear()

                self._ensure_writer(row)
                self._writer.writerow(row)

        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
