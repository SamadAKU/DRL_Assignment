import csv
import os
from typing import Optional


class EpisodeMetricsCSV:
    """
    Minimal, SILENT per-episode CSV logger.
    Collects from VecMonitor infos via the wrapper’s info['metrics'] (if any).

    Columns (written when seen):
      ep_idx, steps, ep_return, ep_len, pellets, deaths, unique_tiles, score_delta, truncations
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        # Create file with header if missing
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ep_idx", "steps", "ep_return", "ep_len",
                    "pellets", "deaths", "unique_tiles", "score_delta",
                    "truncations"
                ])
        self._ep_idx = self._count_existing_rows()

    def _count_existing_rows(self) -> int:
        try:
            with open(self.file_path, "r", newline="") as f:
                # subtract header
                return max(0, sum(1 for _ in f) - 1)
        except FileNotFoundError:
            return 0

    def log_episode(self, info: dict, ep_return: float, ep_len: int) -> None:
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        row = [
            self._ep_idx,
            int(metrics.get("steps", ep_len)),
            float(ep_return),
            int(ep_len),
            int(metrics.get("pellets", 0)),
            int(metrics.get("deaths", 0)),
            int(metrics.get("unique_tiles", 0)),
            float(metrics.get("score_delta", 0.0)),
            int(metrics.get("truncations", 0)),
        ]
        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._ep_idx += 1
