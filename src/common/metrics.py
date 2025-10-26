import os
import csv
import time
from typing import List, Dict
from collections import defaultdict
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np



class EpisodeMetricsLogger(BaseCallback):
    """
    Collect per-episode custom metrics during training and write to CSV.

    - Reads counters from infos[i]['metrics'] (e.g., pellets, deaths, unique_tiles, score_delta, truncations, episode_ale_score).
    - Accumulates over the episode with +=.
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

        # one accumulator dict per env
        self._accum: List[Dict[str, float]] = []
        # track all metric keys we’ve seen so header stays stable
        self._seen_metric_keys: set[str] = set()
        # fallback episode counters per env if the env doesn’t provide episode_number
        self._ep_counter_per_env: List[int] = []

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._accum = [defaultdict(float) for _ in range(n_envs)]
        self._ep_counter_per_env = [0 for _ in range(n_envs)]
        return True

    def _coerce_num(self, v):
        """Convert bool/int/float/numeric-string to float, else None."""
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except ValueError:
                return None
        return None

    def _ensure_writer(self, row: Dict[str, float]) -> None:
        base_cols = ["app", "algo", "persona", "episode", "ep_len", "ep_return", "timestamp_ms"]
        # include all metric keys we’ve seen so far (plus any in this row)
        self._seen_metric_keys.update([k for k in row.keys() if k not in base_cols])
        metric_cols = sorted(self._seen_metric_keys)
        cols = base_cols + metric_cols

        if self._writer is None:
            self._fh = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=cols)
            self._writer.writeheader()
            return

        current = list(self._writer.fieldnames)
        if set(cols) - set(current):
            # expand header: rewrite file preserving old rows
            self._fh.close()
            with open(self._csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
            with open(self._csv_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
            self._fh = open(self._csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._fh, fieldnames=cols)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for i, info in enumerate(infos):
            # 1) accumulate step-delta metrics
            m = info.get("metrics")
            if m:
                acc = self._accum[i]
                for k, v in m.items():
                    num = self._coerce_num(v)
                    if num is not None:
                        acc[k] += num               # <-- accumulate!
                        self._seen_metric_keys.add(k)
                    # non-numeric values are ignored (or handle them separately if you need)

            # 2) on episode end, write a row
            ep = info.get("episode")
            if ep:
                ep_num = int(info.get("episode_number", self._ep_counter_per_env[i]))
                self._ep_counter_per_env[i] = ep_num + 1

                row = {
                    "app": self.app,
                    "algo": self.algo,
                    "persona": self.persona,
                    "episode": ep_num,
                    "ep_len": int(ep.get("l", 0)),
                    "ep_return": float(ep.get("r", 0.0)),
                    "timestamp_ms": int(time.time() * 1000),
                }
                # include all known metric keys; default to 0.0 if never emitted this episode
                for k in self._seen_metric_keys:
                    row[k] = float(self._accum[i].get(k, 0.0))
                self._accum[i].clear()

                self._ensure_writer(row)
                self._writer.writerow(row)
                self._fh.flush()  # helps live graphing

        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
