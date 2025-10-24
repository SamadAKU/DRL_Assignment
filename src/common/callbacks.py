from stable_baselines3.common.callbacks import BaseCallback

class AutoSaveCallback(BaseCallback):
    def __init__(self, save_path, save_freq=1000, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose:
                print(f"Autosaved model to {self.save_path} at step {self.n_calls}")
        return True


from typing import Dict, Any, List, Optional
import csv, os, time
import matplotlib.pyplot as plt

class EpisodeMetricsCSV(BaseCallback):
    """
    Collect per-episode metrics (reward, length, and info['metrics']) across vectorized envs
    and append them to a CSV for training-time analysis.
    """
    def __init__(self, csv_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self._fieldnames: Optional[List[str]] = None
        self._ep_idx = 0
        self._acc = None  # list of dict per env

    def _on_training_start(self) -> None:
        n_envs = getattr(self.training_env, "num_envs", 1)
        self._acc = [ { "ret": 0.0, "len": 0, "totals": {} } for _ in range(n_envs) ]
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        if self.verbose:
            print(f"[EpisodeMetricsCSV] writing to {self.csv_path}")

    def _ensure_writer(self, totals_keys: List[str]):
        # Create header: ep_idx, total_timesteps, ep_len, ep_return, totals...
        if not hasattr(self, "_writer"):
            self._fieldnames = ["ep_idx", "total_timesteps", "ep_len", "ep_return"] + [f"total_{k}" for k in sorted(totals_keys)]
            fresh = not os.path.exists(self.csv_path)
            self._csv_f = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._csv_f, fieldnames=self._fieldnames)
            if fresh:
                self._writer.writeheader()

    def _close_writer(self):
        try:
            self._csv_f.close()
        except Exception:
            pass

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # Vectorized envs
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


class LiveTrainPlotCallback(BaseCallback):
    """
    Periodically reads the EpisodeMetricsCSV and produces the same plots as eval:
      - episode return over index
      - episode length over index
      - totals bar chart (if totals exist)
    Saves PNGs next to the CSV. No GUI. Safe if CSV doesn't exist yet.
    """
    def __init__(self, csv_path: str, plot_freq_steps: int = 50000, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.plot_freq_steps = int(plot_freq_steps)
        self._last_plot = -1

    def _on_step(self) -> bool:
        if self.plot_freq_steps <= 0:
            return True
        if self.num_timesteps - self._last_plot < self.plot_freq_steps:
            return True
        self._last_plot = int(self.num_timesteps)

        try:
            import pandas as pd
            if not os.path.exists(self.csv_path):
                return True
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return True

            base = os.path.splitext(self.csv_path)[0]

            if "ep_return" in df.columns:
                plt.figure()
                plt.plot(df["ep_idx"], df["ep_return"])
                plt.title("Training — Episode Return")
                plt.xlabel("Episode"); plt.ylabel("Return")
                plt.tight_layout()
                p = base + "_ep_return.png"
                plt.savefig(p); plt.close()
                if self.verbose: print("[LiveTrainPlot] saved", p)

            if "ep_len" in df.columns:
                plt.figure()
                plt.plot(df["ep_idx"], df["ep_len"])
                plt.title("Training — Episode Length")
                plt.xlabel("Episode"); plt.ylabel("Steps")
                plt.tight_layout()
                p = base + "_ep_len.png"
                plt.savefig(p); plt.close()
                if self.verbose: print("[LiveTrainPlot] saved", p)

            total_cols = [c for c in df.columns if c.startswith("total_")]
            if total_cols:
                last = df.iloc[-1]
                vals = [float(last[c]) for c in total_cols]
                plt.figure()
                plt.bar(total_cols, vals)
                plt.xticks(rotation=45, ha="right")
                plt.title("Training — Totals (last episode)")
                plt.tight_layout()
                p = base + "_totals.png"
                plt.savefig(p); plt.close()
                if self.verbose: print("[LiveTrainPlot] saved", p)

        except Exception as e:
            if self.verbose:
                print("[LiveTrainPlot] plotting skipped:", e)
        return True
