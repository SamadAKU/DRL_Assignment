# main.py (menu launcher with live training plots + training graph utility)
import os, glob, subprocess, sys

ALGOS = ["ppo", "a2c"]
APPS = ["snake", "pacman"]

def ask(prompt, options):
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        try:
            n = int(input("Enter number: ").strip())
            if 1 <= n <= len(options):
                return options[n-1]
        except ValueError:
            pass

def ask_str(prompt, default=None):
    s = input(f"{prompt}{' ['+default+']' if default else ''}: ").strip()
    return s if s else (default if default is not None else "")

def ask_int(prompt, default=None):
    while True:
        s = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not s and default is not None:
            return int(default)
        try:
            return int(s)
        except ValueError:
            print("Enter an integer.")

def ask_yes_no(prompt, default_yes=True):
    default_txt = "Y/n" if default_yes else "y/N"
    while True:
        s = input(f"{prompt} ({default_txt}): ").strip().lower()
        if not s:
            return default_yes
        if s in ["y","yes"]: return True
        if s in ["n","no"]: return False

def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.call(cmd)

def default_paths(app, persona):
    log_dir   = f"logs/{app}/train"
    ckpt_dir  = f"models/{app}/checkpoints"
    model_dir = f"models/{app}"
    app_cfg   = f"config/app/{app}.yaml"
    reward_cfg= f"config/rewards/{persona}.yaml"
    train_csv = f"logs/{app}/train/episodes.csv"
    return log_dir, ckpt_dir, model_dir, app_cfg, reward_cfg, train_csv

def pick_model(model_dir, app, algo):
    # Prefer *_best.zip, fallback to *_latest.zip, else any zip under models/{app}
    best = sorted(glob.glob(os.path.join(model_dir, f"{algo.upper()}_{app}_best.zip")), reverse=True)
    if best: return best[0]
    latest = sorted(glob.glob(os.path.join(model_dir, f"{algo.upper()}_{app}_latest.zip")), reverse=True)
    if latest: return latest[0]
    anyzip = sorted(glob.glob(os.path.join(model_dir, "*.zip")), reverse=True)
    return anyzip[0] if anyzip else ""

def do_train():
    app = ask("Choose app", APPS)
    algo = ask("Choose algo", ALGOS)
    # persona is just the YAML stem under config/rewards
    reward_yamls = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob("config/rewards/*.yaml") if os.path.isfile(p)])
    persona = ask("Choose persona", reward_yamls) if reward_yamls else ask_str("Persona name (YAML stem)")
    n_envs = ask_int("Vectorized envs", 8)
    timesteps = ask_str("Total timesteps", "3e6")

    live_plots = ask_yes_no("Enable live training plots", True)
    plot_freq  = ask_int("Plot frequency (steps)", 50000) if live_plots else None

    log_dir, ckpt_dir, model_dir, app_cfg, reward_cfg, train_csv = default_paths(app, persona)

    cmd = [
        sys.executable, "-m", "src.train",
        "--app", app,
        "--algo", algo,
        "--persona", persona,
        "--n_envs", str(n_envs),
        "--timesteps", str(timesteps),
        "--log_dir", log_dir,
        "--ckpt_dir", ckpt_dir,
        "--model_dir", model_dir,
        "--app_cfg", app_cfg,
        "--reward_cfg", reward_cfg,
        "--train_metrics_csv", train_csv,
    ]
    if live_plots:
        cmd += ["--live_plots", "--plot_freq", str(plot_freq)]
    run(cmd)
    print("\nTraining launched. CSV & plots will update inside:", log_dir)

def do_watch():
    app = ask("Choose app", APPS)
    algo = ask("Choose algo", ALGOS)
    # persona only used to resolve default cfgs here
    reward_yamls = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob("config/rewards/*.yaml") if os.path.isfile(p)])
    persona = ask("Choose persona (for cfg selection)", reward_yamls) if reward_yamls else ask_str("Persona name")
    _, _, model_dir, app_cfg, reward_cfg, _ = default_paths(app, persona)

    model_path_default = pick_model(model_dir, app, algo)
    model_path = ask_str("Model path (.zip)", model_path_default)
    fps = ask_int("FPS (watch speed)", 30)
    record = ask_str("Record to mp4 (blank = no)", "")

    cmd = [
        sys.executable, "-m", "src.eval",
        "--app", app, "--algo", algo, "--persona", persona,
        "--app_cfg", app_cfg, "--reward_cfg", reward_cfg,
        "--model_path", model_path,
        "--watch", "--fps", str(fps),
    ]
    if record:
        cmd += ["--record", record]
    run(cmd)

def do_eval():
    app = ask("Choose app", APPS)
    algo = ask("Choose algo", ALGOS)
    reward_yamls = sorted([os.path.splitext(os.path.basename(p))[0] for p in glob.glob("config/rewards/*.yaml") if os.path.isfile(p)])
    persona = ask("Choose persona", reward_yamls) if reward_yamls else ask_str("Persona name")
    episodes = ask_int("Episodes", 50)

    log_dir, _, model_dir, app_cfg, reward_cfg, _ = default_paths(app, persona)
    out_csv = ask_str("Out CSV", f"logs/{app}/eval/run1.csv")
    summary_csv = ask_str("Summary CSV", f"logs/{app}/eval/run1_summary.csv")

    model_path_default = pick_model(model_dir, app, algo)
    model_path = ask_str("Model path (.zip)", model_path_default)

    make_plots = ask_yes_no("Make eval plots", True)

    cmd = [
        sys.executable, "-m", "src.eval",
        "--app", app, "--algo", algo, "--persona", persona,
        "--app_cfg", app_cfg, "--reward_cfg", reward_cfg,
        "--model_path", model_path,
        "--episodes", str(episodes),
        "--out_csv", out_csv,
        "--summary_csv", summary_csv,
    ]
    if make_plots:
        cmd.append("--make_plots")
    run(cmd)

def do_graph_training():
    app = ask("Choose app to graph", APPS)
    log_dir = f"logs/{app}/train"
    out = ask_str("Custom output PNG (blank = default)", "")
    cmd = [sys.executable, "-m", "src.plot_training", "--log_dir", log_dir]
    if out:
        cmd += ["--out", out]
    run(cmd)

def main():
    while True:
        choice = ask("What do you want to do?", [
            "Train",
            "Watch a model",
            "Evaluate a model (and make plots)",
            "Graph training logs (VecMonitor)",
            "Quit",
        ])
        if choice.startswith("Train"):
            do_train()
        elif choice.startswith("Watch"):
            do_watch()
        elif choice.startswith("Evaluate"):
            do_eval()
        elif choice.startswith("Graph training"):
            do_graph_training()
        else:
            break

if __name__ == "__main__":
    main()
