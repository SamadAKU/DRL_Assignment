import gym as gym_legacy                  
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation
from gymnasium import spaces as gspaces
from gymnasium import spaces
import numpy as np

# ---- NumPy 2.x compatibility for legacy gym ----
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

class GymV21toGymnasium(gym.Env):
    """
    Wrap a legacy Gym env (reset->obs, step->(obs, reward, done, info))
    and expose Gymnasium API (reset->(obs, info), step->(..., terminated, truncated, info)).
    """
    metadata = {}

    def __init__(self, env):
        super().__init__()
        self.env = env
        # reflect spaces as gymnasium spaces
        self.action_space = self._to_gymnasium_space(env.action_space)
        self.observation_space = self._to_gymnasium_space(env.observation_space)
        self.spec = getattr(env, "spec", None)
        self.metadata = getattr(env, "metadata", {})

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            elif hasattr(self.env, "reset_seed"):
                self.env.reset_seed(seed)
        out = self.env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return obs, (info or {})
        return out, {}

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
            truncated = bool(info.get("TimeLimit.truncated", False))
            terminated = bool(done and not truncated)
            return obs, float(reward), terminated, truncated, info
        # already gymnasium-style
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info

    def render(self):
        if hasattr(self.env, "render"):
            return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            return self.env.close()

    def __getattr__(self, name):
        # pass-through anything else to legacy env
        return getattr(self.env, name)

    def _to_gymnasium_space(self, space):
        # convert legacy gym spaces -> gymnasium spaces
        import numpy as np
        if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape"):
            return gspaces.Box(low=np.array(space.low),
                               high=np.array(space.high),
                               shape=getattr(space, "shape", None),
                               dtype=getattr(space, "dtype", np.float32))
        if hasattr(space, "n") and not hasattr(space, "nvec"):
            return gspaces.Discrete(int(space.n))
        if hasattr(space, "nvec"):
            import numpy as np
            return gspaces.MultiDiscrete(np.array(space.nvec, dtype=np.int64))
        if hasattr(space, "shape") and space.shape != ():
            return gspaces.MultiBinary(space.shape)
        return space

class SnakeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, max_body_parts=100):
        super().__init__(env)
        self.max_body_parts = int(max_body_parts)
        self.grid_size = None
        self.unit_size = None

        d = 12 + 3 * self.max_body_parts
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(d,), dtype=np.float32)
        self._prev_head = None

    def _maybe_init_grid_params(self):
        base = self.unwrapped
        if (self.grid_size is None or self.unit_size is None) and hasattr(base, "controller"):
            grid = base.controller.grid
            self.grid_size = grid.grid_size
            self.unit_size = grid.unit_size

    def _get_food_positions_units(self, base):
        foods = []
        if hasattr(base, "controller"):
            c = base.controller
            if hasattr(c, "foods") and c.foods:
                for f in c.foods:
                    if hasattr(f, "position"):
                        foods.append(np.array(f.position, dtype=np.float32))
                    elif hasattr(f, "pos"):
                        foods.append(np.array(f.pos, dtype=np.float32))
                    elif hasattr(f, "location"):
                        foods.append(np.array(f.location, dtype=np.float32))
        if len(foods) == 0 and hasattr(base, "controller"):
            grid = base.controller.grid
            color = np.array([0, 0, 255], dtype=np.uint8)
            matches = np.all(grid.grid == color, axis=2)
            coords = np.argwhere(matches)
            for y_px, x_px in coords:
                foods.append(np.array([x_px // grid.unit_size, y_px // grid.unit_size], dtype=np.float32))
        return foods

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._maybe_init_grid_params()
        self._prev_head = None
        return self.observation(obs), info

    def observation(self, obs):
        self._maybe_init_grid_params()
        if self.grid_size is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        base = self.unwrapped
        snake = getattr(base.controller, "snakes", [None])[0] if hasattr(base, "controller") else None
        if snake is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        gw, gh = self.grid_size
        norm = np.array([float(gw), float(gh)], dtype=np.float32)

        head = np.array(snake.head, dtype=np.float32)
        head_norm = head / norm

        def _onehot_from_dxdy(dx, dy):
            return np.array([
                1.0 if (dx, dy) == (0, -1) else 0.0,
                1.0 if (dx, dy) == (0,  1) else 0.0,
                1.0 if (dx, dy) == (-1, 0) else 0.0,
                1.0 if (dx, dy) == (1,  0) else 0.0,
            ], dtype=np.float32)

        dir_onehot = np.zeros(4, dtype=np.float32)
        dir_val = getattr(snake, "direction", None)
        used = False
        if dir_val is not None:
            try:
                arr = np.asarray(dir_val)
                if arr.shape == (2,):
                    dx, dy = int(arr[0]), int(arr[1])
                    dir_onehot = _onehot_from_dxdy(dx, dy)
                    used = True
            except Exception:
                pass

        if not used:
            if self._prev_head is None:
                dir_onehot = np.zeros(4, dtype=np.float32)
            else:
                d = head - np.array(self._prev_head, dtype=np.float32)
                if abs(d[0]) > abs(d[1]):
                    dir_onehot = np.array(
                        [0.0, 0.0, 1.0 if d[0] < 0 else 0.0, 1.0 if d[0] > 0 else 0.0],
                        dtype=np.float32
                    )
                else:
                    dir_onehot = np.array(
                        [1.0 if d[1] < 0 else 0.0, 1.0 if d[1] > 0 else 0.0, 0.0, 0.0],
                        dtype=np.float32
                    )

        foods = self._get_food_positions_units(base)
        if len(foods) == 0:
            food_norm = np.array([0.0, 0.0], dtype=np.float32)
            delta_norm = np.array([0.0, 0.0], dtype=np.float32)
            dist_norm = np.array([0.0], dtype=np.float32)
        else:
            foods = np.stack(foods, axis=0)
            dists = np.linalg.norm(foods - head[None, :], axis=1)
            j = int(np.argmin(dists))
            food = foods[j]
            food_norm = food / norm
            delta_norm = (food - head) / norm
            max_d = np.linalg.norm(norm)
            dist_norm = np.array([float(dists[j] / max_d)], dtype=np.float32)

        body_coords = list(snake.body)[-self.max_body_parts:]
        k = len(body_coords)
        if k > 0:
            body_arr = np.stack([np.array(p, dtype=np.float32) for p in body_coords], axis=0) / norm
        else:
            body_arr = np.zeros((0, 2), dtype=np.float32)
        if k < self.max_body_parts:
            pad = np.zeros((self.max_body_parts - k, 2), dtype=np.float32)
            body_arr = np.concatenate([body_arr, pad], axis=0)
        body_flat = body_arr.reshape(-1)

        body_mask = np.zeros((self.max_body_parts,), dtype=np.float32)
        if k > 0:
            body_mask[:k] = 1.0

        length_norm = np.array([min(1.0, k / float(self.max_body_parts))], dtype=np.float32)

        self._prev_head = tuple(head.tolist())

        feat = np.concatenate([
            head_norm,
            dir_onehot,
            food_norm,
            delta_norm,
            dist_norm,
            length_norm,
            body_flat,
            body_mask,
        ]).astype(np.float32)

        return feat

# ------------ Action wrapper: map Discrete(4) to your game's action space ------------
class SnakeActionListWrapper(gym.ActionWrapper):
    """
    Exposes a clean Discrete(4) action-space: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
    and maps to the underlying snake actions.

    If your base env already uses Discrete(4) in that order, this becomes a no-op.
    Otherwise, adjust the mapping table below.
    """
    def __init__(self, env):
        super().__init__(env)
        # public action space for the agent
        self.action_space = gym.spaces.Discrete(4)
        # map our 0..3 to the base env's actions
        # EDIT THIS if your underlying env expects different ints:
        self._map = {
            0: 0,  # UP
            1: 1,  # RIGHT
            2: 2,  # DOWN
            3: 3,  # LEFT
        }

    def action(self, act):
        return self._map[int(act)]

# ------------ Reward+metrics wrapper: shaping and logging during training ------------
class SnakeRewardWrapper(gym.Wrapper):
    """
    Adds reward shaping and per-episode metrics during training.
    It writes info['metrics'] every step so EpisodeMetricsLogger can aggregate.

    reward_cfg (optional):
      {
        "weights": {
          "apple": +1.0,
          "step": -0.01,
          "death": -1.0
        }
      }
    """
    def __init__(self, env, reward_cfg: dict | None = None):
        super().__init__(env)
        w = (reward_cfg or {}).get("weights", {}) if isinstance(reward_cfg, dict) else {}
        self.w_apple = float(w.get("apple", 1.0))
        self.w_step  = float(w.get("step", -0.01))
        self.w_death = float(w.get("death", -1.0))

        # episode accumulators
        self._ep_steps = 0
        self._ep_apples = 0
        self._ep_deaths = 0

        # for simple apple detection if base env doesn't expose it
        self._last_score = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._ep_steps = 0
        self._ep_apples = 0
        # death counter rolls across episodes only on actual game over;
        # ensure no stale metrics make it out on first step
        self._last_score = self._extract_score(info)
        return obs, info

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self._ep_steps += 1

        # --- detect “apple eaten” ---
        apple_flag = self._extract_apple_flag(info, base_r)
        if apple_flag:
            self._ep_apples += 1

        # --- reward shaping ---
        shaped = 0.0
        if apple_flag:
            shaped += self.w_apple
        shaped += self.w_step

        if terminated or truncated:
            # treat any game over as a death (adjust if your env distinguishes)
            self._ep_deaths += 1
            shaped += self.w_death

        reward = float(base_r) + shaped

        # --- per-step metrics for EpisodeMetricsLogger to accumulate ---
        info.setdefault("metrics", {})
        info["metrics"]["steps"] = 1
        info["metrics"]["apples_eaten"] = 1 if apple_flag else 0
        info["metrics"]["deaths"] = 1 if (terminated or truncated) else 0

        return obs, reward, terminated, truncated, info

    # ----- helpers -----
    def _extract_score(self, info):
        # If your env exposes a running score, put the key here (e.g., "score")
        # Returning None falls back to reward-based apple detection.
        return info.get("score")

    def _extract_apple_flag(self, info, base_r):
        # Prefer explicit signal if your env provides it:
        if "apple_eaten" in info:
            return bool(info["apple_eaten"])

        # Next best: score increase
        score = info.get("score")
        if score is not None:
            if self._last_score is None:
                self._last_score = score
                return False
            ate = score > self._last_score
            self._last_score = score
            return ate

        # Fallback: positive base reward => assume apple
        try:
            return float(base_r) > 0
        except Exception:
            return False

def make_snake_env(app: str = "snake", for_watch: bool = False, reward_cfg: dict | None = None):
    import gym_snake  # registers "gym_snake:snake-v0"

    base = gym_legacy.make("gym_snake:snake-v0", disable_env_checker=True)
    env = GymV21toGymnasium(base) 
    if for_watch and hasattr(env.unwrapped, "viewer"):
        try:
            env.unwrapped.viewer = True
        except Exception:
            pass
    env = SnakeActionListWrapper(env)
    env = SnakeObservationWrapper(env, max_body_parts=100)
    env = SnakeRewardWrapper(env, reward_cfg=reward_cfg)
    if isinstance(env.observation_space, gspaces.Box) and len(env.observation_space.shape) > 1:
        from gymnasium.wrappers import FlattenObservation
        env = FlattenObservation(env)
    env = RecordEpisodeStatistics(env)
    return env
