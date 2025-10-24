import argparse, os, csv, time
from typing import Dict, Any
from stable_baselines3 import PPO, A2C
from src.common.factory import make_env

ALGOS = {"ppo": PPO, "a2c": A2C}

def _normalize_model_path(p: str) -> str:
    # Avoid .zip.zip when SB3 appends the suffix internally
    return p[:-4] if p.lower().endswith(".zip") else p

def run_eval_vec(app, algo, persona, app_cfg, reward_cfg, model_path, episodes, out_csv, summary_csv, make_plots):
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    # Vec path (headless). Use DummyVecEnv with 1 env for simplicity & portability.
    def thunk():
        return make_env(app_cfg, reward_cfg, eval_mode=True, render_human=False)
    venv = DummyVecEnv([thunk])
    venv = VecMonitor(venv, os.path.dirname(out_csv) or ".")

    Model = ALGOS[algo]
    model_path = _normalize_model_path(model_path)
    model = Model.load(model_path, env=venv, device="auto")
    
    writer = None
    metric_keys = set()
    totals = {"episodes": 0, "sum_ep_return": 0.0, "sum_ep_len": 0}

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        for ep in range(int(episodes)):
            obs = venv.reset()
            done = False
            ep_len = 0
            ep_ret = 0.0
            agg: Dict[str, float] = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, d, infos = venv.step(action)
                done = bool(d[0])
                ep_len += 1
                ep_ret += float(r[0])
                info = infos[0]
                for k, v in info.get("metrics", {}).items():
                    try:
                        agg[k] = agg.get(k, 0.0) + float(v)
                        metric_keys.add(k)
                    except: pass

            row = {"app": app, "algo": algo, "persona": persona,
                   "episode": ep, "ep_len": ep_len, "ep_return": ep_ret,
                   "timestamp_ms": int(time.time()*1000), **agg}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            totals["episodes"] += 1
            totals["sum_ep_return"] += ep_ret
            totals["sum_ep_len"] += ep_len
            print(f"[eval] ep={ep} ret={ep_ret:.2f} len={ep_len}")

    venv.close()
    _write_summary(out_csv, summary_csv, app, algo, persona, totals, metric_keys)
    if make_plots:
        _make_plots(out_csv, summary_csv, app, algo, persona)

def run_eval_watch(app, algo, persona, app_cfg, reward_cfg, model_path, episodes, out_csv, summary_csv, fps, record_path):
    # Single env, human render on (Pacman shows a window; Snake renders if supported)
    import numpy as np
    try:
        import cv2
        has_cv2 = True
    except Exception:
        has_cv2 = False
    env = make_env(app_cfg, reward_cfg, eval_mode=True, render_human=True)
    Model = ALGOS[algo]
    model_path = _normalize_model_path(model_path)
    model = Model.load(model_path, env=env, device="auto")

    writer = None
    metric_keys = set()
    totals = {"episodes": 0, "sum_ep_return": 0.0, "sum_ep_len": 0}

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    f = open(out_csv, "w", newline="")
    writer = None

    delay = max(1, int(1000 / fps))
    writer_vid = None
    if record_path and not has_cv2:
        print("Recording requested but OpenCV not installed. `pip install opencv-python` to enable.")

    for ep in range(int(episodes)):
        res = env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs, info = res, {}

        done = False
        ep_len = 0
        ep_ret = 0.0
        agg: Dict[str, float] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            ep_len += 1
            ep_ret += float(r)

            # Collect metrics
            for k, v in info.get("metrics", {}).items():
                try:
                    agg[k] = agg.get(k, 0.0) + float(v)
                    metric_keys.add(k)
                except: pass

            # Render (human mode shows its own window), try to capture rgb_array if available
            frame = None
            try:
                frame = env.render()  # Pacman will return an RGB frame even in human mode on some setups
            except Exception:
                pass
            if record_path and has_cv2 and isinstance(frame, np.ndarray):
                if writer_vid is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_vid = cv2.VideoWriter(record_path, fourcc, fps, (w, h))
                # convert RGB->BGR for OpenCV
                writer_vid.write(frame[:, :, ::-1])

            # simple pacing
            if fps > 0:
                # Use cv2.waitKey if available to keep window responsive
                if has_cv2:
                    import cv2
                    if (cv2.waitKey(delay) & 0xFF) == ord('q'):
                        done = True
                else:
                    time.sleep(1.0 / fps)

        row = {"app": app, "algo": algo, "persona": persona,
               "episode": ep, "ep_len": ep_len, "ep_return": ep_ret,
               "timestamp_ms": int(time.time()*1000), **agg}
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
        writer.writerow(row)
        totals["episodes"] += 1
        totals["sum_ep_return"] += ep_ret
        totals["sum_ep_len"] += ep_len
        print(f"[watch] ep={ep} ret={ep_ret:.2f} len={ep_len}")

    f.close()
    if writer_vid is not None:
        writer_vid.release()
    try:
        import cv2
        cv2.destroyAllWindows()
    except Exception:
        pass
    env.close()
    _write_summary(out_csv, summary_csv, app, algo, persona, totals, metric_keys)

def _write_summary(out_csv, summary_csv, app, algo, persona, totals, metric_keys):
    if summary_csv is None:
        summary_csv = out_csv.replace(".csv", "_summary.csv")
    sums: Dict[str, float] = {k: 0.0 for k in metric_keys}
    with open(out_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            for k in metric_keys:
                try:
                    sums[k] += float(r.get(k, 0) or 0)
                except: pass
    summary = {
        "app": app, "algo": algo, "persona": persona,
        "episodes": totals["episodes"],
        "total_return": totals["sum_ep_return"],
        "avg_return": totals["sum_ep_return"] / max(1, totals["episodes"]),
        "total_steps": totals["sum_ep_len"],
        "avg_steps": totals["sum_ep_len"] / max(1, totals["episodes"]),
        **{f"total_{k}": v for k, v in sums.items()},
    }
    os.makedirs(os.path.dirname(summary_csv) or ".", exist_ok=True)
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader(); w.writerow(summary)
    print("Wrote summary ->", summary_csv)

def _make_plots(out_csv, summary_csv, app, algo, persona):
    try:
        import pandas as pd, matplotlib.pyplot as plt
        df = pd.read_csv(out_csv)
        if "ep_return" in df.columns:
            plt.figure(); plt.plot(df.index, df["ep_return"])
            plt.title(f"{app}/{algo}/{persona} - Episode Return"); plt.xlabel("Episode"); plt.ylabel("Return")
            plt.tight_layout(); p = out_csv.replace(".csv", "_ep_return.png"); plt.savefig(p); plt.close(); print("Saved", p)
        if "ep_len" in df.columns:
            plt.figure(); plt.plot(df.index, df["ep_len"])
            plt.title(f"{app}/{algo}/{persona} - Episode Length"); plt.xlabel("Episode"); plt.ylabel("Steps")
            plt.tight_layout(); p = out_csv.replace(".csv", "_ep_len.png"); plt.savefig(p); plt.close(); print("Saved", p)
        import pandas as pd
        sdf = pd.read_csv(summary_csv or out_csv.replace(".csv", "_summary.csv"))
        totals = [c for c in sdf.columns if c.startswith("total_")]
        if totals:
            plt.figure(); plt.bar(totals, [float(sdf.loc[0, c]) for c in totals])
            plt.title(f"{app}/{algo}/{persona} - Totals"); plt.xticks(rotation=45, ha="right")
            plt.tight_layout(); p = (summary_csv or out_csv.replace(".csv", "_summary.csv")).replace(".csv", "_totals.png")
            plt.savefig(p); plt.close(); print("Saved", p)
    except Exception as e:
        print("Plotting failed (ignored):", e)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--app", required=True, choices=["snake", "pacman"])
    p.add_argument("--algo", required=True, choices=["ppo", "a2c"])
    p.add_argument("--persona", required=True)
    p.add_argument("--app_cfg", default=None)
    p.add_argument("--reward_cfg", default=None)
    p.add_argument("--model_path", required=True)
    p.add_argument("--episodes", default="5")
    p.add_argument("--out_csv", required=True)
    p.add_argument("--summary_csv", default=None)
    p.add_argument("--make_plots", action="store_true")

    # NEW flags:
    p.add_argument("--no_vec", action="store_true", help="run single env (no vectorization); good for watching")
    p.add_argument("--watch", action="store_true", help="open a window (Pacman human render); implies --no_vec")
    p.add_argument("--fps", type=int, default=30, help="watch speed (only used with --watch)")
    p.add_argument("--record", default=None, help="mp4 path to record (requires opencv)")

    args = p.parse_args()
    if args.app_cfg is None: args.app_cfg = f"config/app/{args.app}.yaml"
    if args.reward_cfg is None: args.reward_cfg = f"config/rewards/{args.persona}.yaml"

    if args.watch:
        args.no_vec = True

    if args.no_vec:
        run_eval_watch(
            app=args.app, algo=args.algo, persona=args.persona,
            app_cfg=args.app_cfg, reward_cfg=args.reward_cfg,
            model_path=args.model_path, episodes=int(args.episodes),
            out_csv=args.out_csv, summary_csv=args.summary_csv,
            fps=int(args.fps), record_path=args.record,
        )
    else:
        run_eval_vec(
            app=args.app, algo=args.algo, persona=args.persona,
            app_cfg=args.app_cfg, reward_cfg=args.reward_cfg,
            model_path=args.model_path, episodes=int(args.episodes),
            out_csv=args.out_csv, summary_csv=args.summary_csv,
            make_plots=bool(args.make_plots),
        )
