# src/common/callbacks.py — autosave + per-episode CSV logger with metadata
from typing import Optional, List, Dict
import os, csv
from stable_baselines3.common.callbacks import BaseCallback

class AutoSaveCallback(BaseCallback):
    def __init__(self, base_path: str, save_freq: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.base_path = base_path
        self.save_freq = int(save_freq)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            path = f"{self.base_path}.zip"
            if self.verbose:
                print(f"[AutoSave] saving {path}")
            self.model.save(path)
        return True

class EpisodeMetricsCSV(BaseCallback):
    """
    Append per-episode metrics to a CSV during training.
    Writes static metadata columns (app, algo, persona, run_id) on every row.
    """
    def __init__(self, csv_path: str, meta: Optional[Dict[str, str]] = None, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.meta = meta or {}
        self._fieldnames: Optional[List[str]] = None
        self._ep_idx = 0
        self._acc = None  # list of dict per env
        self._csv_f = None
        self._writer = None

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._acc = [ { "ret": 0.0, "len": 0, "totals": {} } for _ in range(n_envs) ]
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if self.verbose:
            print(f"[EpisodeMetricsCSV] writing to {self.csv_path}")

    def _ensure_writer(self, totals_keys: List[str]):
        if not self._writer:
            static_cols = ["app", "algo", "persona", "run_id"]
            self._fieldnames = static_cols + ["ep_idx", "total_timesteps", "ep_len", "ep_return"] + [f"total_{k}" for k in sorted(totals_keys)]
            fresh = not os.path.exists(self.csv_path)
            self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_f, fieldnames=self._fieldnames)
            if fresh:
                self._writer.writeheader()

    def _close_writer(self):
        try:
            if self._csv_f:
                self._csv_f.close()
        except Exception:
            pass
        self._csv_f = None
        self._writer = None

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        n_envs = len(infos)
        for i in range(n_envs):
            info = infos[i] or {}
            metrics = info.get("metrics", None)
            if metrics:
                acc = self._acc[i]["totals"]
                for k, v in metrics.items():
                    acc[k] = acc.get(k, 0.0) + float(v)

            ep = info.get("episode", None)
            if ep is not None:
                self._acc[i]["ret"] = float(ep.get("r", 0.0))
                self._acc[i]["len"] = int(ep.get("l", 0))

                totals_keys = list(self._acc[i]["totals"].keys())
                self._ensure_writer(totals_keys)

                row = {
                    **{k: self.meta.get(k, "") for k in ["app", "algo", "persona", "run_id"]},
                    "ep_idx": self._ep_idx,
                    "total_timesteps": int(self.num_timesteps),
                    "ep_len": self._acc[i]["len"],
                    "ep_return": self._acc[i]["ret"],
                }
                for k in totals_keys:
                    row[f"total_{k}"] = float(self._acc[i]["totals"][k])
                self._writer.writerow(row)

                if self.verbose:
                    print(f"[EpisodeMetricsCSV] wrote episode {self._ep_idx} @ {self.num_timesteps} ts")
                self._ep_idx += 1
                self._acc[i] = { "ret": 0.0, "len": 0, "totals": {} }
        return True

    def _on_training_end(self) -> None:
        self._close_writer()
