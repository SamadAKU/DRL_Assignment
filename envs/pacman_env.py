import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PacmanRewardWrapper(gym.Wrapper):
    """
    Persona-aware shaping + consistent metrics:
      pellets, deaths, unique_tiles, truncations, score_delta, steps

    YAML (reward_cfg["weights"]) can include:
      alive, pellet, death, truncation, explore (or new_tile),
      ghost, fruit

    Exploration uses hashed content of a small obs slice
    (works without reading exact coordinates from ALE RAM).
    """
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        w = (reward_cfg or {}).get("weights", {})

        # core persona knobs
        self.w_alive   = float(w.get("alive",     0.0))
        self.w_pellet  = float(w.get("pellet",    0.0))
        self.w_death   = float(w.get("death",   -10.0))
        self.w_trunc   = float(w.get("truncation", -3.0))
        self.w_explore = float(w.get("explore",   0.0))
        self.w_ghost   = float(w.get("ghost",     0.0))
        self.w_fruit   = float(w.get("fruit",     0.0))

        self._visited = set()
        self._score_delta = 0.0
        self._steps = 0
        self._last_lives = None

    def reset(self, **kwargs):
        res = self.env.reset(**kwargs)
        obs, info = res if isinstance(res, tuple) else (res, {})
        info.setdefault("metrics", {})
        self._visited.clear()
        self._score_delta = 0.0
        self._steps = 0
        self._last_lives = info.get("lives", None)
        return obs, info

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        m = info.setdefault("metrics", {})

        shaped = 0.0

        # alive bonus
        shaped += self.w_alive

        # pellet eaten (positive base reward)
        if base_r > 0:
            shaped += self.w_pellet
            m["pellets"] = m.get("pellets", 0) + 1

        # death detection (drop in lives)
        lives = info.get("lives", self._last_lives)
        if self._last_lives is not None and lives is not None:
            if lives < self._last_lives:
                shaped += self.w_death
                m["deaths"] = m.get("deaths", 0) + 1
        self._last_lives = lives

        # exploration by hashed frame region
        if self.w_explore != 0.0:
            h = int(np.sum(obs[:10, :10]) % (1 << 20))
            if h not in self._visited:
                self._visited.add(h)
                shaped += self.w_explore
                m["unique_tiles"] = m.get("unique_tiles", 0) + 1

        # truncation penalty (time up)
        if truncated and not terminated:
            shaped += self.w_trunc
            m["truncations"] = m.get("truncations", 0) + 1

        # raw gameplay score logging
        self._score_delta += float(base_r)
        m["score_delta"] = self._score_delta
        m["steps"] = self._steps

        return obs, float(base_r + shaped), terminated, truncated, info


def make_pacman_env(reward_cfg, eval_mode: bool = False, render_human: bool = False):
    import ale_py
    render_mode = "human" if render_human else "rgb_array"
    env = gym.make(
        "ALE/MsPacman-v5",
        render_mode=render_mode,
        repeat_action_probability=0.0,
    )
    env = PacmanRewardWrapper(env, reward_cfg)
    return env
