import gymnasium as gym
from gymnasium import spaces
from collections import deque
import numpy as np

class GymV21toGymnasium(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = self._to_gymnasium_space(env.action_space)
        self.observation_space = self._to_gymnasium_space(env.observation_space)
        self.spec = getattr(env, "spec", None)
        self.metadata = getattr(env, "metadata", {})

    def _to_gymnasium_space(self, space):
        if isinstance(space, spaces.Discrete):
            return spaces.Discrete(space.n)
        elif isinstance(space, spaces.Box):
            return spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
        elif isinstance(space, spaces.MultiBinary):
            return spaces.MultiBinary(space.n)
        elif isinstance(space, spaces.MultiDiscrete):
            return spaces.MultiDiscrete(space.nvec)
        else:
            return space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            if hasattr(self.env, "seed"):
                self.env.seed(seed)
            elif hasattr(self.env, "reset_seed"):
                self.env.reset_seed(seed)
        obs = self.env.reset() if options is None else self.env.reset(options=options)
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = result
        return obs, float(reward), terminated, truncated, info

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)


class SnakeActionListWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(4)

    def action(self, act):
        return int(act)


class SnakeRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_cfg=None):
        super().__init__(env)
        self.cfg = reward_cfg or {"weights": {}, "limits": {}}
        w = self.cfg.get("weights", {})

        self.w_step       = float(w.get("step_alive", 0.0))
        self.w_apple      = float(w.get("apple", 2.0))
        self.w_death      = float(w.get("death", -30.0))
        self.w_trunc      = float(w.get("truncation", -5.0))
        self.w_osc        = float(w.get("oscillation", 0.0))
        self.w_new_tile   = float(w.get("new_tile", 0.0))
        self.w_turn_pen   = float(w.get("turn_penalty", 0.0))

        # trackers
        self.prev_distance = None
        self.prev_head = None
        self.steps_since_last_apple = 0
        self.osc_window = int(self.cfg.get("limits", {}).get("osc_window", 4))
        self.recent = deque(maxlen=self.osc_window)
        self.visited = set()   
        self.last_action = None

    def _get_pos(self, grid, color):
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)
        if coords.shape[0] == 0:
            return None
        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def _head_pos(self, grid): return self._get_pos(grid, np.array([255, 0, 0], dtype=np.uint8))
    def _food_pos(self, grid): return self._get_pos(grid, np.array([  0, 0,255], dtype=np.uint8))

    def _dist_to_food(self, grid):
        f, h = self._food_pos(grid), self._head_pos(grid)
        if f is None or h is None: return None
        fx, fy = f; hx, hy = h
        return abs(fx - hx) + abs(fy - hy)

    def _is_oscillating(self):
        if len(self.recent) < 4: return False
        return (self.recent[-1] == self.recent[-3]) and (self.recent[-2] == self.recent[-4])

    def _is_turn(self, action):
        if self.last_action is None: return False
        return action != self.last_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_distance = None
        self.prev_head = None
        self.steps_since_last_apple = 0
        self.recent.clear()
        self.visited.clear()
        self.last_action = None
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        m = info.setdefault("metrics", {})
        m["time_alive"] = m.get("time_alive", 0) + 1
        self.steps_since_last_apple += 1

        if self.w_turn_pen != 0 and self._is_turn(int(action)):
            reward += self.w_turn_pen
            m["turns"] = m.get("turns", 0) + 1
        self.last_action = int(action)

        if reward == 1:
            fast_bonus = max(0.0, 1.0 - 0.02 * self.steps_since_last_apple)
            reward = self.w_apple + fast_bonus
            m["apples_eaten"] = m.get("apples_eaten", 0) + 1

            self.prev_distance = None
            self.prev_head = None
            self.steps_since_last_apple = 0
            self.recent.clear()
            reward += self.w_step
            return obs, float(reward), terminated, truncated, info

        if reward == -1:
            reward = self.w_death
            m["deaths"] = m.get("deaths", 0) + 1

        # Grid-based shaping (distance, oscillation, exploration)
        if hasattr(self.env, "grid") and self.env.grid is not None:
            grid = self.env.grid
            head = self._head_pos(grid)
            if head is not None:
                # exploration: unique tiles
                if self.w_new_tile != 0.0:
                    if head not in self.visited:
                        self.visited.add(head)
                        reward += self.w_new_tile
                        m["unique_tiles"] = m.get("unique_tiles", 0) + 1

                # distance shaping toward food
                dist = self._dist_to_food(grid)
                if dist is not None:
                    if self.prev_distance is not None:
                        delta = self.prev_distance - dist
                        reward += 0.02 * delta 
                    self.prev_distance = dist

                # oscillation penalty
                self.recent.append(head)
                if self._is_oscillating() and self.w_osc != 0.0:
                    reward += self.w_osc
                    m["oscillations"] = m.get("oscillations", 0) + 1

                self.prev_head = head

        # truncation shaping
        if truncated and not terminated:
            reward += self.w_trunc
            m["truncations"] = m.get("truncations", 0) + 1

        # per-step alive shaping
        reward += self.w_step
        return obs, float(reward), terminated, truncated, info
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance = None
        self.prev_head = None
        self.steps_since_last_apple = 0
        self.steps_near_food = 0
        self.osc_penalty = 0.2
        self.recent = deque(maxlen=4)

    def _distance_to_food(self, grid):
        food = self.get_food_position(grid)
        head = self.get_head_position(grid)
        if food is None or head is None:
            return None
        fx, fy = food
        hx, hy = head
        return abs(fx - hx) + abs(fy - hy)

    def get_head_position(self, grid):
        color = np.array([255, 0, 0], dtype=np.uint8)
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)
        if coords.shape[0] == 0:
            return None
        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_food_position(self, grid):
        color = np.array([0, 0, 255], dtype=np.uint8)
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)
        if coords.shape[0] == 0:
            return None
        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps_since_last_apple += 1

        if reward == 1:
            fast_bonus = max(0.0, 1.0 - 0.02 * self.steps_since_last_apple)
            reward = 2.0 + fast_bonus
            self.prev_distance = None
            self.prev_head = None
            self.steps_since_last_apple = 0
            self.steps_near_food = 0
            self.recent.clear()
            return obs, reward, terminated, truncated, info

        if reward == -1:
            reward = -30.0

        if hasattr(self.env, "grid") and self.env.grid is not None:
            grid = self.env.grid

            head = self.get_head_position(grid)
            if head is not None:
                food_pos = self.get_food_position(grid)
                dist = None
                if food_pos is not None:
                    dist = self._distance_to_food(grid)

                if self.prev_head is not None and head == self.prev_head:
                    reward -= 0.05
                else:
                    reward += 0.01

                if dist is not None:
                    if self.prev_distance is not None:
                        delta = self.prev_distance - dist
                        reward += 0.02 * delta
                        if dist <= 2:
                            self.steps_near_food += 1
                            reward += 0.005
                        else:
                            self.steps_near_food = 0
                    else:
                        pass
                else:
                    self.steps_near_food = 0

                self.recent.append(tuple(head.tolist()))
                if len(self.recent) >= 4 and self.recent[-1] == self.recent[-3] and self.recent[-2] == self.recent[-4]:
                    reward -= self.osc_penalty

                self.prev_distance = dist
                self.prev_head = tuple(head.tolist())

        if truncated and not terminated:
            reward -= 5.0

        return obs, float(reward), terminated, truncated, info


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
        if hasattr(self.env, "grid") and self.env.grid is not None:
            self.grid_size = (self.env.grid.width, self.env.grid.height)
            self.unit_size = getattr(self.env.grid, "unit_size", 10)
            return True
        return False

    def observation(self, obs):
        if not self._maybe_init_grid_params():
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        grid = self.env.grid
        food = self._find_color(grid, np.array([0, 0, 255], dtype=np.uint8))
        head = self._find_color(grid, np.array([255, 0, 0], dtype=np.uint8))
        body = self._find_color(grid, np.array([1, 0, 0], dtype=np.uint8))

        head_norm = self._norm_pos(head)
        food_norm = self._norm_pos(food)
        dir_onehot = self._dir_onehot(head)

        delta = (0, 0) if self._prev_head is None else (head[0]-self._prev_head[0], head[1]-self._prev_head[1])
        delta_norm = self._norm_vec(delta)

        dist = 0 if food is None or head is None else abs(food[0]-head[0]) + abs(food[1]-head[1])
        dist_norm = np.array([dist], dtype=np.float32) / float(max(self.grid_size[0], self.grid_size[1], 1))

        length_norm = np.array([len(body)], dtype=np.float32) / float(self.max_body_parts)

        body_flat = np.zeros((self.max_body_parts, 2), dtype=np.float32)
        for i, p in enumerate(body[:self.max_body_parts]):
            body_flat[i] = self._norm_pos(p)[0:2]
        body_flat = body_flat.flatten()

        body_mask = np.zeros((self.max_body_parts,), dtype=np.float32)
        mask_len = min(len(body), self.max_body_parts)
        body_mask[:mask_len] = 1.0

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

    def _find_color(self, grid, color):
        matches = np.all(grid.grid == color, axis=2)
        coords = np.argwhere(matches)
        if coords.shape[0] == 0:
            return None
        y_px, x_px = coords[0]
        return (x_px // grid.unit_size, y_px // grid.unit_size)

    def _norm_pos(self, pos):
        if pos is None or self.grid_size is None:
            return np.zeros((2,), dtype=np.float32)
        w, h = self.grid_size
        x, y = pos
        return np.array([x / max(1, w - 1), y / max(1, h - 1)], dtype=np.float32)

    def _norm_vec(self, v):
        return np.array([v[0], v[1]], dtype=np.float32) / np.array([max(1, self.grid_size[0]), max(1, self.grid_size[1])], dtype=np.float32)

    def _dir_onehot(self, head):
        return np.array([0, 0, 0, 0], dtype=np.float32)

def make_snake_env(reward_cfg, seed=None, eval_mode=False):

    try:
        from gym_snake.envs.snake_env import SnakeEnv
    except ImportError:
        from gym_snake.envs.snake import SnakeEnv

    env = SnakeEnv()
    env = GymV21toGymnasium(env)
    env = SnakeActionListWrapper(env)
    env = SnakeRewardWrapper(env) 
    env = SnakeObservationWrapper(env)

    if seed is not None:
        env.reset(seed=seed)
    return env
