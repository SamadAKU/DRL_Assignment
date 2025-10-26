# envs/snake_env.py
import gym as gym_legacy
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from gymnasium import spaces
import numpy as np

class GymV21toGymnasium(gym.Env):
    """Adapt legacy Gym snake to Gymnasium signatures."""
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space if isinstance(env.action_space, spaces.Space) else spaces.Discrete(getattr(env.action_space, "n", 4))
        self.observation_space = env.observation_space if isinstance(env.observation_space, spaces.Space) else spaces.Box(low=-np.inf, high=np.inf, shape=getattr(env.observation_space, "shape", (1,)), dtype=np.float32)
        self.spec = getattr(env, "spec", None)
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            elif hasattr(self.env, "reset_seed"):
                self.env.reset_seed(seed)
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2: return out
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5: return out
        obs, reward, done, info = out
        return obs, reward, bool(done), False, info

    def render(self, *a, **k): return self.env.render(*a, **k)
    def close(self): return self.env.close()

class SnakeActionListWrapper(gym.ActionWrapper):
    def __init__(self, env, num_actions: int | None = None):
        super().__init__(env)
        n = num_actions if num_actions is not None else getattr(env.action_space, "n", 4)
        self.action_space = spaces.Discrete(int(n))
    def action(self, act): return int(act)

class RenderHumanEveryStep(gym.Wrapper):
    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self.env.render()
        return out
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.env.render()
        return obs, reward, terminated, truncated, info

class SnakeRewardWrapper(gym.Wrapper):
    """
    reward_cfg: {"weights":{"alive","apple","death","truncation","explore"}}
    Emits info['metrics']: steps, apples, deaths, unique_tiles, truncations, score_delta
    """
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        w = (reward_cfg or {}).get("weights", {})
        self.w_alive   = float(w.get("alive",       0.0))
        self.w_apple   = float(w.get("apple",       0.0))
        self.w_death   = float(w.get("death",     -10.0))
        self.w_trunc   = float(w.get("truncation", -3.0))
        self.w_explore = float(w.get("explore",     0.0))
        self._visited = set(); self._score_delta = 0.0; self._steps = 0
        self._last_alive = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info.setdefault("metrics", {})
        self._visited.clear(); self._score_delta = 0.0; self._steps = 0
        self._last_alive = info.get("alive", None)
        return obs, info

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        m = info.setdefault("metrics", {})
        shaped = self.w_alive
        if float(base_r) > 0:
            shaped += self.w_apple
            m["apples"] = m.get("apples", 0) + 1
        alive = info.get("alive", self._last_alive)
        if (self._last_alive is not None) and (alive is not None) and (not bool(alive)) and bool(self._last_alive):
            shaped += self.w_death
            m["deaths"] = m.get("deaths", 0) + 1
        self._last_alive = alive
        if self.w_explore != 0.0:
            arr = np.asarray(obs); sl = arr[:10] if arr.ndim == 1 else arr[:10, :10]
            h = int(np.sum(sl) % (1 << 20))
            if h not in self._visited:
                self._visited.add(h); shaped += self.w_explore
                m["unique_tiles"] = m.get("unique_tiles", 0) + 1
        if truncated and not terminated:
            shaped += self.w_trunc
            m["truncations"] = m.get("truncations", 0) + 1
        self._score_delta += float(base_r)
        m["score_delta"] = self._score_delta
        m["steps"] = self._steps
        return obs, float(base_r + shaped), terminated, truncated, info

def make_snake_env(reward_cfg=None, eval_mode: bool = False, render_human: bool = False):
    env = gym_legacy.make("gym_snake:snake-v0")
    env = GymV21toGymnasium(env)
    if render_human:
        env = RenderHumanEveryStep(env)
    env = SnakeActionListWrapper(env)
    env = SnakeRewardWrapper(env, reward_cfg=reward_cfg)
    env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)  # <-- no deque_size
    return env
