# src/common/factory.py
from __future__ import annotations
from typing import Any, Optional
import os

from envs.pacman_env import make_pacman_env
from envs.snake_env  import make_snake_env
import yaml  

def _load_reward_cfg(reward_cfg_path: Optional[str]) -> Optional[dict]:
    if reward_cfg_path is None:
        return None
    if isinstance(reward_cfg_path, dict):
        return reward_cfg_path
    if isinstance(reward_cfg_path, str) and reward_cfg_path.strip():
        path = reward_cfg_path
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            return {}
        return data
    return None

def make_env(app: str, for_watch: bool = False, reward_cfg_path=None):
    key = app.strip().lower()

    if key in {"snake", "snake-v0", "gym_snake:snake-v0"}:
        return make_snake_env(app="snake", for_watch=for_watch, reward_cfg= _load_reward_cfg(reward_cfg_path))

    if key in {"pacman", "ms_pacman", "mspacman"}:
        return make_pacman_env(for_watch=for_watch, reward_cfg_path=reward_cfg_path)

    raise ValueError(f"Unknown app '{app}'")
