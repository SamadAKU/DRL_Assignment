# Packman/wrapper.py
from typing import Optional, Dict, Any
import gymnasium as gym
import numpy as np

# Remove YAML if not needed — leave support here for personas later
try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


class Controls(gym.Wrapper):
    """
    Restrict actions to UP/DOWN/LEFT/RIGHT for DRL stability
    """
    def __init__(self, env):
        super().__init__(env)
        import gymnasium.spaces as spaces
        self.action_space = spaces.Discrete(4)
        # Minimal actions: UP, DOWN, RIGHT, LEFT
        self._map = np.array([1, 4, 2, 3], dtype=np.int64)

    def step(self, a):
        return self.env.step(self._map[int(a)])

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class PacmanRewardWrapper(gym.Wrapper):
    """
    Custom reward shaping:
    - +score * scale
    - +survival bonus each timestep
    - -penalty for too long no score
    - -big penalty on death
    """
    def __init__(self, env, cfg: Optional[Dict[str, Any]] = None, yaml_path: Optional[str] = None):
        super().__init__(env)

        if cfg is None and yaml_path and _HAS_YAML:
            try:
                with open(yaml_path, "r") as f:
                    cfg = yaml.safe_load(f)
            except Exception:
                cfg = {}

        cfg = cfg or {}
        self.score_scale = float(cfg.get("score_scale", 1.0))
        self.survive_bonus = float(cfg.get("survive_bonus", 0.1))
        self.no_score_penalty = float(cfg.get("no_score_penalty", -0.01))
        self.no_score_patience = int(cfg.get("no_score_patience", 50))
        self.death_penalty = float(cfg.get("death_penalty", -20.0))

        self._since_score = 0

    def reset(self, **kwargs):
        self._since_score = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        shaped = reward * self.score_scale

        # Survival: encourage progress searching
        shaped += self.survive_bonus

        if reward > 0:
            self._since_score = 0
        else:
            self._since_score += 1

        if self._since_score > self.no_score_patience:
            shaped += self.no_score_penalty

        # Big penalty if killed
        if done:
            shaped += self.death_penalty

        return obs, shaped, done, truncated, info


def make_rewarded_env(env, yaml_path: Optional[str] = None):
    """
    Called automatically by factory → applies mapping + shaping
    """
    env = Controls(env)
    env = PacmanRewardWrapper(env, yaml_path=yaml_path)
    return env
