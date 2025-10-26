import argparse
import os
from typing import Tuple, Any

import numpy as np
from stable_baselines3 import PPO, A2C

from src.common.factory import make_env

ALGOS = {"ppo": PPO, "a2c": A2C}


def _load(algo: str, path: str):
    a = algo.lower()
    if a == "ppo":
        return PPO.load(path, device="auto")
    if a == "a2c":
        return A2C.load(path, device="auto")
    raise ValueError("algo must be ppo or a2c")


def _safe_render(env) -> None:
    """
    Be tolerant of legacy gym/game renderers:
    - Some return an RGB frame, some open a window, some do nothing.
    - We just call it; if it returns a frame, we ignore it.
    """
    try:
        frame = env.render()
        _ = frame  # ignore; user just wants to watch
    except TypeError:
        # Some legacy envs require a mode argument
        try:
            frame = env.render(mode="human")
            _ = frame
        except Exception:
            pass
    except Exception:
        # Don't let rendering kill watching
        pass


def watch(app: str, algo: str, model_path: str, episodes: int):
    # Build the same env stack you use in training, but with for_watch=True (no reward shaping)
    env = make_env(app, for_watch=True, reward_cfg_path=None)

    # SB3 models can predict on a raw (non-Vec) env just fine
    model = _load(algo, model_path)

    for ep in range(episodes):
        reset_out = env.reset()
        # Gymnasium reset is (obs, info); tolerate legacy (obs) just in case
        if isinstance(reset_out, tuple) and len(reset_out) == 2:
            obs, info = reset_out
        else:
            obs, info = reset_out, {}

        ep_rew = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Model expects obs shaped like during training; wrappers in make_env handle this
            action, _ = model.predict(obs, deterministic=True)
            step_out = env.step(action)
            # Gymnasium: (obs, reward, terminated, truncated, info)
            if len(step_out) == 5:
                obs, reward, terminated, truncated, info = step_out
            else:
                # Legacy: (obs, reward, done, info) -> treat done as terminated
                obs, reward, done, info = step_out
                terminated = bool(done)
                truncated = False

            ep_rew += float(reward)
            _safe_render(env)

        print(f"[WATCH] {app} episode={ep+1}/{episodes} return={ep_rew:.2f}")

    try:
        env.close()
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["pacman", "snake"])
    ap.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    # Basic sanity checks
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")

    watch(args.app, args.algo, args.model_path, args.episodes)


if __name__ == "__main__":
    main()
