import argparse, os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

from src.common.factory import make_env
from src.common.callbacks import AutoSaveCallback, EpisodeMetricsCSV, LiveTrainPlotCallback

ALGOS = {"ppo": PPO, "a2c": A2C}

def build_vec_env(app_cfg, reward_cfg, n_envs, eval_mode=False):
    def thunk(_rank):
        def _init():
            return make_env(app_cfg, reward_cfg, eval_mode=eval_mode)
        return _init
    return SubprocVecEnv([thunk(i) for i in range(n_envs)], start_method="spawn")

def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    venv = build_vec_env(args.app_cfg, args.reward_cfg, args.n_envs, eval_mode=False)
    venv = VecMonitor(venv, args.log_dir)

    Model = ALGOS[args.algo]
    latest_path = os.path.join(args.model_dir, f"{args.algo}_{args.app}_latest")
    final_path  = os.path.join(args.model_dir, f"{args.algo}_{args.app}_final")

    if os.path.exists(latest_path + ".zip"):
        model = Model.load(latest_path, env=venv, device="auto")
        print("Resumed from latest autosave")
    else:
        model = Model("MlpPolicy", venv, verbose=1, tensorboard_log=args.log_dir)

    ckpt = CheckpointCallback(save_freq=args.ckpt_freq, save_path=args.ckpt_dir,
                              name_prefix=f"{args.algo}_{args.app}")
    autosave = AutoSaveCallback(save_path=latest_path, save_freq=args.autosave_freq, verbose=1)

    model.learn(total_timesteps=int(float(args.timesteps)), callback=[ckpt, autosave])
    model.save(final_path)
    venv.close()
    print("Saved final ->", final_path + ".zip")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, choices=["snake", "pacman"])
    p.add_argument("--algo", default="ppo", choices=["ppo", "a2c"])
    p.add_argument("--persona", required=True)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--timesteps", default="3e6")
    p.add_argument("--ckpt_freq", type=int, default=100_000)
    p.add_argument("--autosave_freq", type=int, default=10_000)
    p.add_argument("--app_cfg", default=None)
    p.add_argument("--reward_cfg", default=None)
    p.add_argument("--log_dir", default="logs/{app}/train")
    p.add_argument("--ckpt_dir", default="models/{app}/checkpoints")
    p.add_argument("--model_dir", default="models/{app}")
    p.add_argument("--train_metrics_csv", default="logs/{app}/train/episodes.csv")
    p.add_argument("--live_plots", action="store_true")
    p.add_argument("--plot_freq", type=int, default=50000)
    args = p.parse_args()

    # resolve template paths
    args.log_dir   = args.log_dir.format(app=args.app)
    args.ckpt_dir  = args.ckpt_dir.format(app=args.app)
    args.model_dir = args.model_dir.format(app=args.app)
    args.train_metrics_csv = args.train_metrics_csv.format(app=args.app)
    if args.reward_cfg is None:
        args.reward_cfg = f"config/rewards/{args.persona}.yaml"
    if args.app_cfg is None:
        args.app_cfg = f"config/app/{args.app}.yaml"

    main(args)
