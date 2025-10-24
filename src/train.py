import argparse, os, time
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from src.common.factory import make_env
from src.common.callbacks import AutoSaveCallback, EpisodeMetricsCSV

def build_vec_env(app_cfg, reward_cfg, n_envs, eval_mode=False, render_human=False):
    def _init(rank):
        def _thunk():
            return make_env(app_cfg, reward_cfg, eval_mode=eval_mode, render_human=render_human)
        return _thunk
    venv = SubprocVecEnv([_init(i) for i in range(n_envs)], start_method="spawn")
    venv = VecMonitor(venv)         # writes monitor.csv (gitignored)
    venv = VecTransposeImage(venv)  # (C,H,W) for CNN policies
    return venv

def main(args):
    venv = build_vec_env(args.app_cfg, args.reward_cfg, args.n_envs, eval_mode=False)

    algo = args.algo.lower()
    model_cls = PPO if algo == "ppo" else A2C
    # verbose=1 prints only SB3 tables; 0 would hide even those
    model = model_cls("CnnPolicy", venv, verbose=1, tensorboard_log=args.log_dir)

    # Ensure dirs exist
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.train_metrics_csv), exist_ok=True)

    run_id = time.strftime("%Y%m%d-%H%M%S")

    ckpt = CheckpointCallback(
        save_freq=int(args.ckpt_freq),
        save_path=args.ckpt_dir,
        name_prefix=f"{args.algo.upper()}_{args.app}"
    )
    autosave = AutoSaveCallback(
        os.path.join(args.model_dir, f"{args.algo.upper()}_{args.app}_latest"),
        save_freq=int(args.autosave_freq),
        verbose=0  # silence autosave prints
    )
    epcsv = EpisodeMetricsCSV(
        args.train_metrics_csv,
        meta={"app": args.app, "algo": args.algo, "persona": args.persona, "run_id": run_id},
        verbose=0  # silence CSV prints (still writes fully)
    )

    # Single concise line about outputs
    print(f"[train] logs -> {args.log_dir} | ckpts -> {args.ckpt_dir} | model dir -> {args.model_dir} | csv -> {args.train_metrics_csv}")

    try:
        model.learn(
            total_timesteps=int(float(args.timesteps)),
            callback=[ckpt, autosave, epcsv]
        )
    except KeyboardInterrupt:
        # Save a snapshot and exit cleanly on Ctrl+C
        latest = os.path.join(args.model_dir, f"{args.algo.upper()}_{args.app}_interrupt")
        model.save(latest)
        print(f"\n[train] Interrupted. Saved snapshot to {latest}.")
        return

    model.save(os.path.join(args.model_dir, f"{args.algo.upper()}_{args.app}_final"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True)
    p.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    p.add_argument("--persona", required=True)
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--timesteps", default="3e6")
    # per-persona paths
    p.add_argument("--log_dir", default="logs/{app}/{persona}/train")
    p.add_argument("--ckpt_dir", default="models/{app}/{persona}/checkpoints")
    p.add_argument("--model_dir", default="models/{app}/{persona}")
    p.add_argument("--app_cfg", required=True)
    p.add_argument("--reward_cfg", required=True)
    p.add_argument("--ckpt_freq", type=int, default=100000)
    p.add_argument("--autosave_freq", type=int, default=50000)
    p.add_argument("--train_metrics_csv", default="logs/{app}/{persona}/train/episodes.csv")
    args = p.parse_args()

    # expand placeholders
    args.log_dir = args.log_dir.format(app=args.app, persona=args.persona)
    args.ckpt_dir = args.ckpt_dir.format(app=args.app, persona=args.persona)
    args.model_dir = args.model_dir.format(app=args.app, persona=args.persona)
    args.train_metrics_csv = args.train_metrics_csv.format(app=args.app, persona=args.persona)

    main(args)
