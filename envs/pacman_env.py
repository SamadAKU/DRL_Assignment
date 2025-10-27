from __future__ import annotations
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation
from gymnasium import spaces
import numpy as np
import yaml
from src.common.wrappers import SimpleFrameStack
import os        
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


def make_pacman_env(
    *, 
    for_watch: bool = False,
    reward_cfg_path: str | None = None,
    # legacy kwargs kept for backwards-compat (safe to ignore if unused elsewhere)
    reward_cfg: dict | None = None,
    eval_mode: bool = False,
    render_human: bool | None = None,
):
    """Unified Pacman env factory used by src.common.factory.make_env.
    - `for_watch` toggles human rendering.
    - `reward_cfg_path` (yaml) is loaded and passed to PacmanRewardWrapper.
    - legacy args (`reward_cfg`, `render_human`) still work.
    """
    # Windows spawn: make sure ALE registers in each subprocess
    import ale_py  # noqa: F401

    # prefer for_watch unless caller explicitly passed render_human
    if render_human is None:
        render_human = bool(for_watch)

    # load reward cfg if a path is provided
    if reward_cfg is None and reward_cfg_path and os.path.exists(reward_cfg_path):
        with open(reward_cfg_path, "r") as f:
            reward_cfg = yaml.safe_load(f)

    # In watch mode, disable shaping unless caller explicitly provided nonzero weights
    if for_watch:
        # If reward_cfg is empty or None, force zeros to ensure pure base rewards
        w = (reward_cfg or {}).get("weights", {})
        if not any(abs(float(v)) > 0.0 for v in w.values()) if isinstance(w, dict) else True:
            reward_cfg = {"weights": {"alive": 0.0, "pellet": 0.0, "death": 0.0, "truncation": 0.0, "explore": 0.0}}

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
    env = RecordEpisodeStatistics(env)
    return env
