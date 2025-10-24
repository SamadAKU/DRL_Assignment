# src/common/factory.py
import gymnasium as gym
import gym as gym_legacy
from pacman.wrapper import  Controls, PacmanRewardWrapper
from snake.wrapper import GymV21toGymnasium, SnakeActionListWrapper, SnakeRewardWrapper

from gymnasium.wrappers import RecordEpisodeStatistics, FlattenObservation
from src.common.wrappers import SimpleFrameStack

def make_env(app: str, for_watch: bool = False, reward_cfg_path=None):
    key = app.strip().lower()

    # ----------------------------------------------------
    # PAC-MAN (RAM, MLP, your exact wrapper chain)
    # ----------------------------------------------------
    if key in {"pacman", "ms_pacman", "mspacman"}:

        # Ensure ALE is registered
        import ale_py  # noqa: F401

        render_mode = "human" if for_watch else None

        env = gym.make(
            "ALE/MsPacman-v5",
            obs_type="ram",
            frameskip=4,
            repeat_action_probability=0.0,
            render_mode=render_mode,
        )

        # Your wrapper chain, exactly as original
        env = Controls(env)
        env = SimpleFrameStack(env, num_stack=4, axis=-1)
        env = PacmanRewardWrapper(env)

        env = FlattenObservation(env)
        env = RecordEpisodeStatistics(env)
        return env

    # ----------------------------------------------------
    # SNAKE (legacy gym)
    # ----------------------------------------------------
    if key in {"snake", "snake-v0", "gym_snake:snake-v0"}:
        # Accept either spelling from menu
        snake_id = "gym_snake:snake-v0" if ":" in key else "snake-v0"
        env = gym_legacy.make(snake_id)

        env = GymV21toGymnasium(env)
        env = SnakeActionListWrapper(env)
        env = SnakeRewardWrapper(env)

        env = FlattenObservation(env)

        try:
            env = RecordEpisodeStatistics(env)
        except Exception:
            pass

        return env

    raise ValueError(f"Unknown app '{app}'. Use pacman or snake.")
