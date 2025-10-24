import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
try:
    from envs.snake_env import GymV21toGymnasium 
except Exception:
    GymV21toGymnasium = None

class PacmanRewardWrapper(gym.Wrapper):
    """
    Persona-aware shaping + per-step metrics via info['metrics'].

    We read weights from reward_cfg['weights'] if provided:
      step_alive, pellet, death, truncation, ghost_eaten, new_tile

    We do not try to reverse-engineer all MsPacman scoring events; instead:
      - use env's raw reward as a scoring delta (score_delta)
      - count pellets heuristically via positive deltas
      - penalize death via lives drop
      - encourage exploration via unique tile visits
    """
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        w = (reward_cfg or {}).get("weights", {})
        self.w_alive   = float(w.get("step_alive", 0.0))
        self.w_pellet  = float(w.get("pellet", 0.0))
        self.w_ghost   = float(w.get("ghost_eaten", 0.0))  # kept for persona parity
        self.w_death   = float(w.get("death", -25.0))
        self.w_trunc   = float(w.get("truncation", -5.0))
        self.w_newtile = float(w.get("new_tile", 0.0))

        self._visited = set()
        self._last_lives = None
        self._steps = 0

class PacmanRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        self.reward_cfg = reward_cfg or {}
        raw = (self.reward_cfg.get("weights") or {})

        # normalize keys, add a couple of sensible aliases
        aliases = {
            "living": "alive",
            "power": "power_pellet",
            "new_tile": "newtile",
            "newtiles": "newtile",
            "cherries": "fruit",
        }
        weights = {}
        for k, v in raw.items():
            k_norm = aliases.get(k, k)
            weights[k_norm] = float(v)

        # install all weights as attributes: weights["pellet"] -> self.w_pellet
        for k, v in weights.items():
            setattr(self, f"w_{k}", v)

        # ensure anything step() might access exists with a safe default
        must_have = [
            "alive", "pellet", "power_pellet", "fruit",
            "ghost_chain", "explore", "death", "level_clear",
            "newtile" 
        ]
        for name in must_have:
            if not hasattr(self, f"w_{name}"):
                setattr(self, f"w_{name}", 0.0)

        # map explore → newtile if YAML only gave explore
        if getattr(self, "w_newtile", None) is None or not hasattr(self, "w_newtile"):
            setattr(self, "w_newtile", getattr(self, "w_explore", 0.0))

        # episode state your step() might use
        self._visited = set()
        self._last_lives = None
        self._last_score = 0
        self._steps = 0



    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}
        self._visited.clear()
        self._steps = 0
        if not isinstance(info, dict):
            info = {}
        info.setdefault("metrics", {})   
        self._last_score = info.get("score", 0) if "score" in info else 0
        self._last_lives = info.get("lives", None)
        return obs, info

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        m = info.setdefault("metrics", {})
        m["steps"] = m.get("steps", 0) + 1

        shaped = 0.0
        # Alive shaping
        shaped += self.w_alive

        # Positive reward → likely pellet/fruit/etc.
        if base_r > 0:
            shaped += self.w_pellet
            m["pellets"] = m.get("pellets", 0) + 1

        # Detect death via lives drop (Gymnasium Atari puts 'lives' in info)
        lives = info.get("lives", self._last_lives)
        if self._last_lives is not None and lives is not None and lives < self._last_lives:
            shaped += self.w_death
            m["deaths"] = m.get("deaths", 0) + 1
        self._last_lives = lives

        # Exploration: approximate position from RAM not exposed; instead
        # derive a coarse tile from pixel location of the player is complex.
        # Simple fallback: encourage time-based unique step positions by hashing observation hash.
        # (Keeps it deterministic per frame content; light-weight signal.)
        if self.w_newtile != 0.0:
            # hash a small slice to reduce cost
            h = int(np.sum(obs[:10, :10]) % (1 << 20))
            if h not in self._visited:
                self._visited.add(h)
                shaped += self.w_newtile
                m["unique_tiles"] = m.get("unique_tiles", 0) + 1

        # Truncation shaping
        if truncated and not terminated:
            shaped += self.w_trunc
            m["truncations"] = m.get("truncations", 0) + 1

        # Log raw score delta for analysis
        m["score_delta"] = m.get("score_delta", 0.0) + float(base_r)

        return obs, float(base_r + shaped), terminated, truncated, info


def make_pacman_env(reward_cfg, eval_mode: bool = False, render_human: bool = False):
    import ale_py  # ensure ALE present
    render_mode = "human" if render_human else "rgb_array"

    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode=render_mode,
        full_action_space=False,
        repeat_action_probability=0.0,
    )
    env = PacmanRewardWrapper(env, reward_cfg=reward_cfg)
    return env
