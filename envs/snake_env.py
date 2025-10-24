# envs/snake_env.py  (drop-in fix)

import gym as gym_legacy
from gymnasium.wrappers import FlattenObservation, RecordEpisodeStatistics
from snake.wrapper import (
    GymV21toGymnasium,
    SnakeActionListWrapper,
    SnakeRewardWrapper,
)

def make_snake_env():
    env = gym_legacy.make("snake-v0")  # some installs register this id

    env = GymV21toGymnasium(env)       # legacy Gym -> Gymnasium-like API
    env = SnakeActionListWrapper(env)  # scalar action -> list if your env needs it
    env = SnakeRewardWrapper(env)      # your shaping (no YAML, no fallback)

    # MLP expects a 1D vector
    env = FlattenObservation(env)

    # Episode stats (skip silently if legacy API leaks through)
    try:
        env = RecordEpisodeStatistics(env)
    except Exception:
        pass

    return env
