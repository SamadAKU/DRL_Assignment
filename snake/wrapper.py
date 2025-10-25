from typing import Optional, Dict, Any
import gym as gym_legacy
import numpy as np

try:
    import yaml
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


class GymV21toGymnasium(gym_legacy.Wrapper):
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs, {}
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        truncated = False
        return obs, reward, done, truncated, info


class SnakeActionListWrapper(gym_legacy.Wrapper):
    def step(self, action):
        return self.env.step([int(action)])


class SnakeRewardWrapper(gym_legacy.Wrapper):
    """
    Reward shaping + per-episode metrics in info["metrics"].
    - +1 apple
    - small + for moving closer to food
    - small - for moving away
    - big - on death
    """
    def __init__(self, env, cfg: Optional[Dict[str, Any]] = None, yaml_path: Optional[str] = None):
        super().__init__(env)

        if cfg is None and yaml_path and _HAS_YAML:
            try:
                with open(yaml_path, "r") as f:
                    cfg = yaml.safe_load(f)
            except:
                cfg = {}

        cfg = cfg or {}
        self.apple_bonus = float(cfg.get("apple_bonus", 5.0))
        self.toward_food = float(cfg.get("toward_food", 0.2))
        self.away_from_food = float(cfg.get("away_from_food", -0.2))
        self.death_penalty = float(cfg.get("death_penalty", -20.0))

        self._prev_head = None
        self._steps = 0
        self._food = 0
        self._deaths = 0

    def reset(self, **kwargs):
        self._prev_head = None
        self._steps = 0
        self._food = 0
        self._deaths = 0

        res = self.env.reset(**kwargs)
        obs, info = res if isinstance(res, tuple) else (res, {})
        info.setdefault("metrics", {})
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        shaped = 0.0

        food = info.get("food")
        head = info.get("head")
        if food is not None and head is not None:
            prev_dist = np.linalg.norm(np.subtract(self._prev_head, food)) if self._prev_head is not None else None
            curr_dist = np.linalg.norm(np.subtract(head, food))
            if prev_dist is not None:
                shaped += self.toward_food if curr_dist < prev_dist else self.away_from_food
            self._prev_head = np.array(head)

        if reward > 0:
            shaped += self.apple_bonus

        if done:
            shaped += self.death_penalty

        # --- metrics for training logs ---
        self._steps += 1
        m = info.setdefault("metrics", {})
        m["steps"] = self._steps

        if reward > 0:
            self._food += 1
            m["apples_eaten"] = m.get("apples_eaten", 0) + 1  # rename to "food_eaten" if you prefer

        if done and not truncated:
            self._deaths += 1
            m["deaths"] = m.get("deaths", 0) + 1

        return obs, shaped, done, truncated, info


def make_rewarded_env(env, yaml_path: Optional[str] = None):
    env = GymV21toGymnasium(env)
    env = SnakeActionListWrapper(env)
    env = SnakeRewardWrapper(env, yaml_path=yaml_path)
    return env
