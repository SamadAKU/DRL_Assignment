import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces


class AddChannelWrapper:
    """
    Ensure obs is channel-first (C,H,W).
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_chw(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._to_chw(obs), reward, terminated, truncated, info

    def render(self, *a, **k):
        return self.env.render(*a, **k)

    def close(self):
        return self.env.close()

    def _to_chw(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 2:
            return arr[None, :, :]
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  
            return arr
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4): 
            return np.transpose(arr, (2, 0, 1))
        return arr


class SimpleFrameStack(gym.Wrapper):
    """
    Gymnasium-compatible frame stacker.
    Stacks along `axis` (default last).
    """
    def __init__(self, env, num_stack: int = 4, axis: int = -1):
        super().__init__(env)
        self.k = int(num_stack)
        self.axis = int(axis)
        self.frames = deque(maxlen=self.k)

        orig_space = env.observation_space
        if isinstance(orig_space, spaces.Box):
            shape = tuple(orig_space.shape)
            axis = self.axis if self.axis >= 0 else (len(shape) + self.axis)
            new_shape = list(shape)
            new_shape[axis] = new_shape[axis] * self.k
            new_shape = tuple(new_shape)
            self.observation_space = spaces.Box(
                low=np.min(orig_space.low),
                high=np.max(orig_space.high),
                shape=new_shape,
                dtype=orig_space.dtype,
            )
        else:
            self.observation_space = orig_space

    def _stack(self):
        return np.concatenate(list(self.frames), axis=self.axis)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        arr = np.asarray(obs)
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(arr.copy())
        return self._stack(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(np.asarray(obs))
        return self._stack(), reward, terminated, truncated, info
