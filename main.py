# main.py — menu with auto-discovery of apps, personas, and models (no typing required)
import os
import glob
import subprocess
import sys
from pathlib import Path
import csv

APPS = ["pacman", "snake"]
ALGOS = ["ppo", "a2c"]

# ---------- FS helpers ----------
def merge_monitor_csvs(log_dir: str) -> str:
    """
    Merge all monitor_*.csv files in log_dir into monitor_merged.csv
    (keeps the header from the first file, skips others).
    Returns the merged file path.
    """
    p = Path(log_dir)
    files = sorted(p.glob("monitor_*.csv"))
    if not files:
        one = p / "monitor.csv"
        return str(one) if one.exists() else ""

    out_path = p / "monitor_merged.csv"
    with out_path.open("w", newline="") as fout:
        writer = None
        for i, f in enumerate(files):
            with f.open("r", newline="") as fin:
                reader = csv.reader(fin)
                header = next(reader, None)
                if header is None:
                    continue
                if writer is None:
                    writer = csv.writer(fout)
                    writer.writerow(header)
                # skip header for subsequent files
                for row in reader:
                    if row:
                        writer.writerow(row)
    return str(out_path)


def stems(paths):
    out = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        if name:
            out.append(name)
    return out

def existing_personas_for_app(app: str):
    """Union of personas from config/rewards/*.yaml and models/<app>/* (excluding checkpoints)."""
    cfg_personas = set(stems(glob.glob("config/rewards/*.yaml")))
    model_dirs = set()
    for p in glob.glob(f"models/{app}/*"):
        base = os.path.basename(p)
        if os.path.isdir(p) and base.lower() != "checkpoints":
            model_dirs.add(base)
    return sorted(cfg_personas | model_dirs)

def available_models(app: str, persona: str):
    """
    Look for *best.zip / *latest.zip under:
      logs/<app>/<persona>/<algo>/
      logs/<app>/<persona>/<algo>/checkpoints/
    Return: {algo: {"best": path or "", "latest": path or ""}}
    """
    root = Path(f"logs/{app}/{persona}")
    out: dict[str, dict[str, str]] = {}
    if not root.exists():
        return out

    for algo_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        
        candidates = [algo_dir, algo_dir / "checkpoints"]
        best_path = ""
        latest_path = ""
        for c in candidates:
            if not c.exists():
                continue
            
            for z in c.glob("*.zip"):
                name = z.name.lower()
                if "best" in name:
                    best_path = str(z)
                elif "latest" in name:
                    latest_path = str(z)
        if best_path or latest_path:
            out[algo_dir.name.lower()] = {"best": best_path, "latest": latest_path}
    return out

def pick(items, title):
    """Numbered picker. Returns selected item (string)."""
    if not items:
        raise RuntimeError(f"No options for {title}")
    print(f"\n{title}")
    for i, it in enumerate(items, 1):
        print(f"  [{i}] {it}")
    while True:
        s = input("Choose #: ").strip()
        if not s:
            continue
        try:
            n = int(s)
            if 1 <= n <= len(items):
                return items[n - 1]
        except ValueError:
            pass

def prompt_int(title, default):
    s = input(f"{title} [{default}]: ").strip()
    return int(s) if s else int(default)

def prompt_str(title, default=""):
    s = input(f"{title}{' ['+default+']' if default else ''}: ").strip()
    return s if s else default

def run(cmd):
    cmd = [str(c) for c in cmd]
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, check=False)


def paths_for(app: str, persona: str):
    log_dir   = f"logs/{app}/{persona}/train"
    ckpt_dir  = f"models/{app}/{persona}/checkpoints"
    model_dir = f"models/{app}/{persona}"
    reward_cfg = f"config/rewards/{persona}.yaml"  
    return log_dir, ckpt_dir, model_dir, reward_cfg


def action_train():
    app = pick(APPS, "App")
    algo = pick(ALGOS, "Algorithm")
    personas = existing_personas_for_app(app)
    if personas:
        persona = pick(personas, "Persona (from configs/models)")
    else:
        persona = prompt_str("Persona name (used for save path)")
    timesteps = prompt_str("Total timesteps", "3000000")
    n_envs = prompt_int("Vectorized envs", 8)

    log_dir, ckpt_dir, model_dir, reward_cfg = paths_for(app, persona)

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "src.train",
        "--app", app, "--algo", algo,
        "--timesteps", timesteps, "--n_envs", n_envs,
        "--persona", persona,
    ]
    if os.path.exists(reward_cfg):
        cmd += ["--reward_cfg", reward_cfg]
    run(cmd)

def action_watch():
    app = pick(APPS, "App")
    algo = pick(ALGOS, "Algorithm")
    personas = existing_personas_for_app(app)
    if not personas:
        print("No personas found. Train first.")
        return
    persona = pick(personas, "Persona")

    models = available_models(app, persona)
   
    model_path = ""
    if algo in models:
        model_path = models[algo]["best"] or models[algo]["latest"]
    if not model_path:
        # check any algo present
        for a, rec in models.items():
            model_path = rec["best"] or rec["latest"]
            if model_path:
                print(f"(Note) Using {a.upper()} model because {algo.upper()} not found.")
                algo = a
                break

    if not model_path:
        print("No trained model found for that persona. Train first.")
        return

    episodes = prompt_int("Episodes to watch", 5)
    cmd = [
        sys.executable, "-m", "src.test",
        "--app", app, "--algo", algo,
        "--model_path", model_path, "--episodes", episodes,
    ]
    run(cmd)

def action_plot_single():
    app = pick(APPS, "App")
    persona = pick(existing_personas_for_app(app), "Persona")

    log_root = Path("logs") / app / persona
    if not log_root.exists():
        print("No logs yet for this persona.")
        return

 
    train_csvs = list(log_root.rglob("episodes_train.csv"))
    episodes_csv = None
    if train_csvs:
        episodes_csv = str(max(train_csvs, key=lambda p: p.stat().st_mtime))

 
    if not episodes_csv:
        episodes_csv = merge_monitor_csvs(log_root)  

 
    if not episodes_csv:
        prog = list(log_root.rglob("progress.csv"))
        if prog:
            episodes_csv = str(max(prog, key=lambda p: p.stat().st_mtime))

    if not episodes_csv:
        print("No monitor or training CSVs found. Train first.")
        return

 
    ep_path = Path(episodes_csv)
    summary_csv = ep_path.with_name("progress.csv")
    if not summary_csv.exists():
        # scan up to find nearest progress.csv
        progs = list(ep_path.parent.rglob("progress.csv"))
        if progs:
            summary_csv = max(progs, key=lambda p: p.stat().st_mtime)
        else:
            summary_csv = None

    outdir = ep_path.parent / "plots"
    outdir.mkdir(parents=True, exist_ok=True)

    if os.path.exists("notebooks/plot_results.py"):
        cmd = [sys.executable, "notebooks/plot_results.py", "--episodes_csv", str(ep_path),
               "--outdir", str(outdir)]
        if summary_csv:
            cmd += ["--summary_csv", str(summary_csv)]
        run(cmd)
    else:
        print("Missing notebooks/plot_results.py")
        print("Use this episodes CSV:", str(ep_path))

def action_plot_explorer_survivor():
    script = "notebooks/plot_explorer_survival_comparisons.py"
    if not os.path.exists(script):
        print("Missing:", script)
        return
    cmd = [sys.executable, script]
    run(cmd)

def action_plot_compare():
    app = pick(APPS, "App")
    personas = existing_personas_for_app(app)
    if len(personas) < 2:
        print("Need at least two personas to compare.")
        return
    print("\nPick personas to compare (space-separated indices):")
    for i, p in enumerate(personas, 1):
        print(f"  [{i}] {p}")
    idx = input("Indices (e.g., 1 3 4): ").strip().split()
    sel = []
    for s in idx:
        try:
            n = int(s)
            if 1 <= n <= len(personas):
                sel.append(personas[n-1])
        except ValueError:
            pass
    if len(sel) < 2:
        print("Select at least two.")
        return
    metric = prompt_str("Metric (x-axis key)", "ep_return")

    script_candidates = [
        "notebooks/plot_comare.py",     
        "notebooks/compare_results.py",  
    ]
    script = next((s for s in script_candidates if os.path.exists(s)), "")
    if not script:
        print("Missing compare plot script. Selected personas:", sel)
        return

    out = f"logs/{app}/compare_{metric}.png"
    cmd = [sys.executable, script, "--app", app, "--metric", metric, "--out", out, "--personas", *sel]
    run(cmd)
    print("Wrote:", out)

def main():
    MENU = [
        "Watch (play current model)",
        "Train",
        "Plot single persona (Rewards)",
        "Compare personas (Data Analytics)",
        "Quit",
    ]
    while True:
        choice = pick(MENU, "Main Menu")
        if choice.startswith("Watch"):
            action_watch()
        elif choice == "Train":
            action_train()
        elif choice.startswith("Plot single"):
            action_plot_single()
        elif choice.startswith("Compare personas"):
            action_plot_explorer_survivor()
        else:
            print("bye")
            break

if __name__ == "__main__":
    main()
