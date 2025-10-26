import argparse
import os
import time
from pathlib import Path

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList

from src.common.factory import make_env
from src.common.callbacks import (
    ProgressPrinter,
    LatestModelSaver,
    BestModelSaver,
    PeriodicCheckpointSaver,
)
from src.common.metrics import EpisodeMetricsLogger  # <-- custom metrics

ALGOS = {"ppo": PPO, "a2c": A2C}


def _persona_from_cfg(path: str | None) -> str:
    if not path:
        return "default"
    p = Path(path)
    name = p.stem
    return name


def build_vec_env(app: str, reward_cfg_path: str | None, n_envs: int, log_dir_algo: str):
    def thunk():
        return make_env(app, for_watch=False, reward_cfg_path=reward_cfg_path)
    venv = SubprocVecEnv([thunk for _ in range(n_envs)])
    venv = VecMonitor(venv, log_dir_algo)
    return venv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, choices=["pacman", "snake"])
    p.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--reward_cfg", default=None)
    p.add_argument("--persona", default=None)
    args = p.parse_args()

    app = args.app
    algo = args.algo
    persona = args.persona or _persona_from_cfg(args.reward_cfg)

    # Folder paths
    log_root = Path("logs") / app / persona
    log_dir_algo = log_root / algo.lower() 
    ckpt_dir = log_dir_algo / "checkpoints"

    log_dir_algo.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Vec env
    venv = build_vec_env(app, args.reward_cfg, args.n_envs, str(log_dir_algo))

    # Algo
    Algo = ALGOS[algo.lower()]
    model = Algo("MlpPolicy", venv, verbose=1)

    # Callbacks
    callback_list = CallbackList([
        ProgressPrinter(print_freq=50_000),
        LatestModelSaver(save_dir=str(ckpt_dir), filename=f"{algo}_{app}_{persona}_latest"),
        BestModelSaver(save_dir=str(ckpt_dir), filename=f"{algo}_{app}_{persona}_best"),
        PeriodicCheckpointSaver(save_dir=str(ckpt_dir), save_freq=1_000_000,
                                prefix=f"{algo}_{app}_{persona}"),
        EpisodeMetricsLogger(str(log_dir_algo), app, algo.lower(), persona),
    ])

    # Train
    model.learn(total_timesteps=int(args.timesteps), callback=callback_list)

    # Final latest save
    latest_path = log_dir_algo / f"{algo}_{app}_{persona}_latest_final"
    model.save(str(latest_path))

    venv.close()


if __name__ == "__main__":
    main()
