# main.py — menu launcher (per-persona dirs, quiet console)
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

def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd))
    subprocess.call(cmd)

def default_paths(app, persona):
    log_dir   = f"logs/{app}/{persona}/train"
    ckpt_dir  = f"models/{app}/{persona}/checkpoints"
    model_dir = f"models/{app}/{persona}"
    app_cfg   = f"config/app/{app}.yaml"
    reward_cfg= f"config/rewards/{persona}.yaml"
    train_csv = f"{log_dir}/episodes.csv"
    return log_dir, ckpt_dir, model_dir, app_cfg, reward_cfg, train_csv

def pick_model(model_dir, app, algo):
    best = sorted(glob.glob(os.path.join(model_dir, f"{algo.upper()}_{app}_best.zip")), reverse=True)
    if best: return best[0]
    latest = sorted(glob.glob(os.path.join(model_dir, f"{algo.upper()}_{app}_latest.zip")), reverse=True)
    if latest: return latest[0]
    anyzip = sorted(glob.glob(os.path.join(model_dir, "*.zip")), reverse=True)
    return anyzip[0] if anyzip else ""

def do_train():
    app = ask("Choose app", APPS)
    algo = ask("Choose algo", ALGOS)

    reward_yamls = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob("config/rewards/*.yaml")
                          if os.path.isfile(p))
    prefix = "snake_" if app == "snake" else "pacman_"
    reward_yamls = [y for y in reward_yamls if y.startswith(prefix)]
    persona = ask("Choose persona", reward_yamls) if reward_yamls else ask_str("Persona name (YAML stem)")

    n_envs = ask_int("Vectorized envs", 8)
    timesteps = ask_str("Total timesteps", "3e6")

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
    run(cmd)
    print("\nTraining running. CSV & checkpoints are under:", log_dir, "and", ckpt_dir)

def do_watch():
    app = ask("Choose app", APPS)
    algo = ask("Choose algo", ALGOS)

    reward_yamls = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob("config/rewards/*.yaml")
                          if os.path.isfile(p))
    prefix = "snake_" if app == "snake" else "pacman_"
    reward_yamls = [y for y in reward_yamls if y.startswith(prefix)]
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

    reward_yamls = sorted(os.path.splitext(os.path.basename(p))[0]
                          for p in glob.glob("config/rewards/*.yaml")
                          if os.path.isfile(p))
    prefix = "snake_" if app == "snake" else "pacman_"
    reward_yamls = [y for y in reward_yamls if y.startswith(prefix)]
    persona = ask("Choose persona", reward_yamls) if reward_yamls else ask_str("Persona name")

    episodes = ask_int("Episodes", 50)
    log_dir, _, model_dir, app_cfg, reward_cfg, _ = default_paths(app, persona)
    out_csv = ask_str("Out CSV", f"logs/{app}/{persona}/eval/run1.csv")
    summary_csv = ask_str("Summary CSV", f"logs/{app}/{persona}/eval/run1_summary.csv")
    model_path_default = pick_model(model_dir, app, algo)
    model_path = ask_str("Model path (.zip)", model_path_default)
    make_plots = ask("Make eval plots?", ["Yes", "No"]) == "Yes"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

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

def main():
    while True:
        choice = ask("What do you want to do?", [
            "Train",
            "Watch a model",
            "Evaluate a model (and make plots)",
            "Quit",
        ])
        if choice == "Train":
            do_train()
        elif choice.startswith("Watch"):
            do_watch()
        elif choice.startswith("Evaluate"):
            do_eval()
        else:
            break

if __name__ == "__main__":
    main()
