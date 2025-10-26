from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation
from gymnasium import spaces
import numpy as np

from src.common.wrappers import SimpleFrameStack

class Controls(gym.ActionWrapper):
    """Map Discrete(4) -> ALE MsPacman actions (UP,DOWN,LEFT,RIGHT)."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)
        self._ale_map = self._build_map()

    def _build_map(self):
        default = {"UP": 2, "DOWN": 5, "LEFT": 4, "RIGHT": 3}
        get_meanings = getattr(getattr(self.env, "unwrapped", self.env), "get_action_meanings", None)
        if callable(get_meanings):
            meanings = list(get_meanings())
            want = ["UP", "DOWN", "LEFT", "RIGHT"]
            have = {}
            for name in want:
                if name in meanings:
                    have[name] = meanings.index(name)
                else:
                    idxs = [i for i, m in enumerate(meanings) if name in m]
                    if idxs:
                        have[name] = idxs[0]
            for k, v in default.items():
                if k not in have:
                    have[k] = v
            return (have["UP"], have["DOWN"], have["LEFT"], have["RIGHT"])
        return (default["UP"], default["DOWN"], default["LEFT"], default["RIGHT"])

    def action(self, a: int):
        up, down, left, right = self._ale_map
        if a == 0: return up
        if a == 1: return down
        if a == 2: return left
        return right

class PacmanRewardWrapper(gym.Wrapper):
    """
    reward_cfg: {"weights": {"alive","pellet","death","truncation","explore"}}
    Emits info["metrics"]: pellets, deaths, unique_tiles, truncations, score_delta, steps
    """
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        w = (reward_cfg or {}).get("weights", {})
        self.w_alive   = float(w.get("alive",      0.0))
        self.w_pellet  = float(w.get("pellet",     0.0))
        self.w_death   = float(w.get("death",    -10.0))
        self.w_trunc   = float(w.get("truncation", -3.0))
        self.w_explore = float(w.get("explore",    0.0))
        self._visited = set()
        self._score_delta = 0.0
        self._steps = 0
        self._last_lives = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
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
        shaped += self.w_alive

        if base_r > 0:
            shaped += self.w_pellet
            m["pellets"] = m.get("pellets", 0) + 1

        lives = info.get("lives", self._last_lives)
        if (self._last_lives is not None) and (lives is not None) and (lives < self._last_lives):
            shaped += self.w_death
            m["deaths"] = m.get("deaths", 0) + 1
        self._last_lives = lives

        if self.w_explore != 0.0:
            arr = np.asarray(obs)
            sl = arr[:10] if arr.ndim == 1 else arr[:10, :10]
            h = int(np.sum(sl) % (1 << 20))
            if h not in self._visited:
                self._visited.add(h)
                shaped += self.w_explore
                m["unique_tiles"] = m.get("unique_tiles", 0) + 1

        if truncated and not terminated:
            shaped += self.w_trunc
            m["truncations"] = m.get("truncations", 0) + 1

        self._score_delta += float(base_r)
        m["score_delta"] = self._score_delta
        m["steps"] = self._steps

        return obs, float(base_r + shaped), terminated, truncated, info

class AleScoreTracker(gym.Wrapper):
    """Track true ALE reward; writes metrics['ale_score'] and ['episode_ale_score'] at done."""
    def __init__(self, env):
        super().__init__(env)
        self._ale_score = 0.0

    def reset(self, **kwargs):
        self._ale_score = 0.0
        return super().reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ale_score += float(reward)
        m = info.setdefault("metrics", {})
        m["ale_score"] = self._ale_score
        if terminated or truncated:
            m["episode_ale_score"] = self._ale_score
        return obs, reward, terminated, truncated, info

def make_pacman_env(reward_cfg=None, eval_mode: bool = False, render_human: bool = False):
    # Ensure ALE registration in each worker on Windows
    import ale_py

    env = gym.make(
        "ALE/MsPacman-v5",
        obs_type="ram",
        frameskip=4,
        repeat_action_probability=0.0,
        render_mode=("human" if render_human else None),
    )
    env = Controls(env)
    env = SimpleFrameStack(env, num_stack=4, axis=-1)
    env = PacmanRewardWrapper(env, reward_cfg=reward_cfg)
    env = AleScoreTracker(env)
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)  # <-- no deque_size (compat with your Gymnasium)
    return env
