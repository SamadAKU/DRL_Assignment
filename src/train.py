# src/train.py
import argparse
from pathlib import Path

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from src.common.factory import make_env
from src.common.callbacks import ProgressPrinter, LatestModelSaver, BestModelSaver, PeriodicCheckpointSaver

def _persona_from_cfg(path: str) -> str:
    if not path:
        return "default"
    p = Path(path)
    name = p.name
    return name[:-5] if name.endswith(".yaml") else name

def build_vec_env(app: str, n_envs: int, reward_cfg: str | None):
    def thunk():
        return make_env(app, for_watch=False, reward_cfg_path=reward_cfg)
    if n_envs > 1:
        env = SubprocVecEnv([thunk for _ in range(n_envs)])
    else:
        env = DummyVecEnv([thunk])
    env = VecMonitor(env)
    return env

def make_model(algo: str, env):
    a = algo.lower().strip()
    if a == "ppo":
        return PPO("MlpPolicy", env, verbose=1)
    if a == "a2c":
        return A2C("MlpPolicy", env, verbose=1)
    raise ValueError("algo must be ppo or a2c")

def train(app: str, algo: str, timesteps: int, n_envs: int, reward_cfg: str | None, persona: str | None):
    if not persona:
        persona = _persona_from_cfg(reward_cfg or "")

    model_dir = Path(f"models/{app}/{persona}")
    log_dir = Path(f"logs/{app}/{persona}/train")
    ckpt_dir = model_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = build_vec_env(app, n_envs=n_envs, reward_cfg=reward_cfg)
    model = make_model(algo, env)

    latest = str(model_dir / f"{algo.upper()}_{app}_{persona}_latest.zip")
    best   = str(model_dir / f"{algo.upper()}_{app}_{persona}_best.zip")

    callbacks = [
        ProgressPrinter(print_freq=50000),
        LatestModelSaver(save_path=latest, save_freq=max(100000 // max(n_envs,1), 10000)),
        BestModelSaver(save_path=best),
        PeriodicCheckpointSaver(folder=str(ckpt_dir), save_freq=500000, prefix=f"{algo}_{app}_{persona}"),
    ]

    model.learn(total_timesteps=int(float(timesteps)), callback=callbacks)
    model.save(latest)

    try:
        env.close()
    except Exception:
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, choices=["pacman", "snake"])
    p.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    p.add_argument("--timesteps", type=str, required=True)  # allow "3e6"
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--reward_cfg", type=str, default=None)
    p.add_argument("--persona", type=str, default=None)
    a = p.parse_args()

    train(
        app=a.app,
        algo=a.algo,
        timesteps=int(float(a.timesteps)),
        n_envs=a.n_envs,
        reward_cfg=a.reward_cfg,
        persona=a.persona,
    )

if __name__ == "__main__":
    main()
