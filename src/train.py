import argparse
from pathlib import Path
import os

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from src.common.factory import make_env
from src.common.callbacks import (
    ProgressPrinter,
    LatestModelSaver,
    BestModelSaver,
    PeriodicCheckpointSaver,
    EpisodeMetricsLogger,   # logs custom per-episode metrics during training
)

ALGOS = {"ppo": PPO, "a2c": A2C}


def _persona_from_cfg(path: str | None) -> str:
    if not path:
        return "default"
    p = Path(path)
    name = p.name
    return name[:-5] if name.endswith(".yaml") else name


def build_vec_env(app: str, reward_cfg_path: str | None, n_envs: int):
    def thunk():
        # MATCH your factory signature; do NOT pass eval_mode/render_human
        return make_env(app, for_watch=False, reward_cfg_path=reward_cfg_path)
    if n_envs > 1:
        return SubprocVecEnv([thunk for _ in range(n_envs)])
    return DummyVecEnv([thunk])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, choices=["pacman", "snake"])
    p.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    p.add_argument("--timesteps", type=int, default=3_000_000)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--app_cfg", default=None)         # kept for compatibility if you use it elsewhere
    p.add_argument("--reward_cfg", default=None)
    p.add_argument("--persona", default=None, help="Optional override; otherwise from --reward_cfg name")
    p.add_argument("--log_root", default="logs")
    p.add_argument("--print_freq", type=int, default=50_000)
    p.add_argument("--save_freq", type=int, default=1_000_000)
    args = p.parse_args()

    app = args.app
    algo = args.algo
    timesteps = int(args.timesteps)
    n_envs = int(args.n_envs)

    persona = args.persona or _persona_from_cfg(args.reward_cfg)

    # Paths
    log_dir = Path(args.log_root) / app / persona / algo
    ckpt_dir = log_dir / "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    # Vec env + monitors
    venv = build_vec_env(app=args.app, reward_cfg_path=args.reward_cfg, n_envs=args.n_envs)
    venv = VecMonitor(venv, filename=str(log_dir / "progress.csv"))

    # Algo
    Algo = ALGOS[algo]
    model = Algo("MlpPolicy", venv, verbose=0)

    # Callbacks (EpisodeMetricsLogger writes episodes_train.csv with apples/pellets/etc.)
    callback = [
        ProgressPrinter(print_freq=args.print_freq),
        LatestModelSaver(save_dir=str(ckpt_dir), filename=f"{algo}_{app}_{persona}_latest"),
        BestModelSaver(save_dir=str(ckpt_dir), filename=f"{algo}_{app}_{persona}_best"),
        PeriodicCheckpointSaver(save_dir=str(ckpt_dir), save_freq=args.save_freq, prefix=f"{algo}_{app}_{persona}"),
        EpisodeMetricsLogger(log_dir=str(log_dir), app=app, algo=algo, persona=persona),
    ]

    model.learn(total_timesteps=timesteps, callback=callback)
    venv.close()


if __name__ == "__main__":
    main()
