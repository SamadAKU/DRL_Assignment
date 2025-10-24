# src/common/wrappers.py
import numpy as np
from collections import deque
import gymnasium as gym
from gymnasium import spaces

class AddChannelWrapper:
    """
    Ensure obs is channel-first (C,H,W).
    Handles Gymnasium (reset->(obs,info), step->5-tuple) and legacy Gym.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = getattr(env, "observation_space", None)
        self.metadata = getattr(env, "metadata", {})
        self.spec = getattr(env, "spec", None)

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return self._to_chw(obs), info
        return self._to_chw(out)

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, done, truncated, info = out
            return self._to_chw(obs), reward, done, truncated, info
        else:
            obs, reward, done, info = out
            return self._to_chw(obs), reward, done, info

    def render(self, *a, **k):
        return self.env.render(*a, **k)

    def close(self):
        return self.env.close()

    def _to_chw(self, obs):
        arr = np.asarray(obs)
        if arr.ndim == 2:
            return arr[None, :, :]
        if arr.ndim == 3 and arr.shape[0] in (1,3,4):  # already C,H,W
            return arr
        if arr.ndim == 3 and arr.shape[-1] in (1,3,4): # H,W,C -> C,H,W
            return np.transpose(arr, (2,0,1))
        return arr


class SimpleFrameStack(gym.Wrapper):
    """
    Gymnasium-compatible frame stacker.
    - Stacks observations along `axis` (default last).
    - Updates observation_space so SB3 builds the right MLP input size.
    - Works for vector RAM (e.g., (128,)) and images (HWC/CHW).
    """
    def __init__(self, env, num_stack: int = 4, axis: int = -1):
        super().__init__(env)
        self.k = int(num_stack)
        self.axis = int(axis)
        self.frames = deque(maxlen=self.k)

        # --- build new observation_space ---
        orig_space = env.observation_space
        if not isinstance(orig_space, spaces.Box):
            # Fallback: let FlattenObservation handle weird spaces
            self.observation_space = orig_space
        else:
            shape = tuple(orig_space.shape)
            if len(shape) == 0:
                # scalar → repeat k times
                new_shape = (self.k,)
            else:
                axis = self.axis if self.axis >= 0 else (len(shape) + self.axis)
                # expand the stacking axis by k (vector last-axis default)
                new_shape = list(shape)
                new_shape[axis] = new_shape[axis] * self.k
                new_shape = tuple(new_shape)

            # Repeat bounds along that axis
            low = orig_space.low
            high = orig_space.high
            # Ensure arrays
            low = np.asarray(low)
            high = np.asarray(high)

            # Broadcast-then-concat k times along axis
            lows = [low] * self.k
            highs = [high] * self.k
            new_low = np.concatenate(lows, axis=self.axis) if low.size else low
            new_high = np.concatenate(highs, axis=self.axis) if high.size else high

            self.observation_space = spaces.Box(
                low=new_low.min() if new_low.shape != new_shape else new_low,
                high=new_high.max() if new_high.shape != new_shape else new_high,
                shape=new_shape,
                dtype=orig_space.dtype,
            )

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
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, terminated, info = out
            truncated = False
        self.frames.append(np.asarray(obs))
        return self._stack(), reward, terminated, truncated, info
