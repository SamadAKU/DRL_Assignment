# envs/pacman_env.py
import gymnasium as gym
import numpy as np
from typing import Optional, Dict, Any, Tuple

# Some setups import this from snake_env; keep a safe fallback so this file stands alone.
try:
    from envs.snake_env import GymV21toGymnasium  # noqa: F401
except Exception:
    GymV21toGymnasium = None  # type: ignore


class PacmanRewardWrapper(gym.Wrapper):
    """
    Persona-aware shaping + per-step metrics via info['metrics'].

    Reads weights from reward_cfg['weights'] if provided. Recognized keys (aliases allowed):
      - step_alive (alias: living -> alive)
      - pellet
      - power_pellet (alias: power)
      - fruit (alias: cherries)
      - ghost_chain  (for bonus when chain-eating ghosts, not detected here but kept for parity)
      - new_tile (aliases: newtiles, explore -> newtile)
      - death
      - truncation
      - level_clear

    We do not try to decode all MsPacman events; instead:
      - use env's raw reward as a score delta (logged under 'score_delta')
      - heuristically count pellets on positive base reward
      - detect 'death' via lives drop
      - encourage exploration via a cheap 'unique tile' hash
      - expose per-step counters in info['metrics'] so a CSV callback can sum to total_*
    """

    def __init__(self, env: gym.Env, reward_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(env)
        self.reward_cfg = reward_cfg or {}
        raw = (self.reward_cfg.get("weights") or {})

        # normalize keys, add useful aliases
        aliases = {
            "living": "alive",
            "power": "power_pellet",
            "new_tile": "newtile",
            "newtiles": "newtile",
            "explore": "newtile",
            "cherries": "fruit",
        }
        weights: Dict[str, float] = {}
        for k, v in raw.items():
            k_norm = aliases.get(k, k)
            weights[k_norm] = float(v)

        # install all weights as attributes: weights["pellet"] -> self.w_pellet
        for k, v in weights.items():
            setattr(self, f"w_{k}", v)

        # ensure attributes exist with safe defaults
        for name in [
            "alive", "pellet", "power_pellet", "fruit",
            "ghost_chain", "newtile", "death", "level_clear", "truncation"
        ]:
            if not hasattr(self, f"w_{name}"):
                setattr(self, f"w_{name}", 0.0)

        # episode state
        self._visited = set()
        self._last_lives: Optional[int] = None
        self._steps = 0

    # -------- gym API --------
    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}
        if not isinstance(info, dict):
            info = {}
        info.setdefault("metrics", {})

        self._visited.clear()
        self._steps = 0
        self._last_lives = info.get("lives", None)

        # Optional: initialize counters on reset so eval can read zeros immediately
        m = info["metrics"]
        m.setdefault("steps", 0)
        m.setdefault("pellets", 0)
        m.setdefault("unique_tiles", 0)
        m.setdefault("ghosts_eaten", 0)
        m.setdefault("deaths", 0)
        m.setdefault("truncations", 0)
        m.setdefault("score_delta", 0.0)
        return obs, info

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1

        # make sure we have a metrics dict
        info = dict(info) if isinstance(info, dict) else {}
        m: Dict[str, Any] = info.setdefault("metrics", {})
        m["steps"] = m.get("steps", 0) + 1

        shaped = 0.0

        # 1) alive shaping every step
        if getattr(self, "w_alive", 0.0) != 0.0:
            shaped += float(self.w_alive)

        # 2) positive reward heuristic -> pellets/fruit (count once per positive delta)
        if base_r > 0:
            if getattr(self, "w_pellet", 0.0) != 0.0:
                shaped += float(self.w_pellet)
            m["pellets"] = m.get("pellets", 0) + 1

        # 3) exploration: cheap content hash for "new tile"
        w_newtile = getattr(self, "w_newtile", 0.0)
        if w_newtile != 0.0:
            # Hash a small observation slice to approximate novelty cheaply.
            try:
                # Assume obs is HWC (Gymnasium Atari); slice top-left patch
                h = int(np.sum(obs[:10, :10]) % (1 << 20))
                if h not in self._visited:
                    self._visited.add(h)
                    shaped += float(w_newtile)
                    m["unique_tiles"] = m.get("unique_tiles", 0) + 1
            except Exception:
                # If obs shape unexpected, just skip novelty
                pass

        # 4) death detection via lives drop
        lives = info.get("lives", self._last_lives)
        if self._last_lives is not None and lives is not None and lives < self._last_lives:
            if getattr(self, "w_death", 0.0) != 0.0:
                shaped += float(self.w_death)
            m["deaths"] = m.get("deaths", 0) + 1
        self._last_lives = lives

        # 5) truncation shaping (time limit etc.)
        if truncated and not terminated:
            if getattr(self, "w_truncation", 0.0) != 0.0:
                shaped += float(self.w_truncation)
            m["truncations"] = m.get("truncations", 0) + 1

        # 6) always log score delta for analysis
        m["score_delta"] = float(m.get("score_delta", 0.0) + float(base_r))

        return obs, float(base_r + shaped), terminated, truncated, info


def make_pacman_env(reward_cfg: Optional[Dict[str, Any]], eval_mode: bool = False, render_human: bool = False):
    import ale_py  # noqa: F401

    render_mode = "human" if render_human else "rgb_array"
    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode=render_mode,
        full_action_space=False,
        repeat_action_probability=0.0, 
    )
    env = PacmanRewardWrapper(env, reward_cfg=reward_cfg)
    return env
