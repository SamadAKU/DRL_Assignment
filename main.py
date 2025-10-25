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
    """Return {'best': path or '', 'latest': path or ''} for any algo found."""
    base = Path(f"models/{app}/{persona}")
    found = {}
    if not base.exists():
        return found
    for z in base.glob("*.zip"):
        found.setdefault(z.stem.split("_")[0].lower(), {"best": "", "latest": ""})
    for algo in list(found.keys()):
        best = base / f"{algo.upper()}_{app}_{persona}_best.zip"
        latest = base / f"{algo.upper()}_{app}_{persona}_latest.zip"
        found[algo]["best"] = str(best) if best.exists() else ""
        found[algo]["latest"] = str(latest) if latest.exists() else ""
    return found

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


# ---------- Paths ----------
def paths_for(app: str, persona: str):
    log_dir   = f"logs/{app}/{persona}/train"
    ckpt_dir  = f"models/{app}/{persona}/checkpoints"
    model_dir = f"models/{app}/{persona}"
    reward_cfg = f"config/rewards/{persona}.yaml"  # optional; may not exist
    return log_dir, ckpt_dir, model_dir, reward_cfg

# ---------- Actions ----------
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
    # Prefer the selected algo; fall back if no files yet
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
    log_dir, *_ = paths_for(app, persona)

    # NEW: prefer training metrics if present
    train_csv = Path(log_dir, "episodes_train.csv")
    if train_csv.exists():
        episodes_csv = str(train_csv)
    else:
        episodes_csv = merge_monitor_csvs(log_dir)  # your existing function
        if not episodes_csv:
            # also allow progress.csv as a fallback for quick plots
            prog = Path(log_dir, "progress.csv")
            if prog.exists():
                episodes_csv = str(prog)
            else:
                print("No monitor CSVs found. Train first.")
                return

    summary_csv = str(Path(log_dir, "progress.csv"))

    if os.path.exists("notebooks/plot_results.py"):
        run([sys.executable, "notebooks/plot_results.py",
             "--episodes_csv", episodes_csv,
             "--summary_csv", summary_csv,
             "--outdir", str(Path(log_dir, "plots"))])
    else:
        print("Missing notebooks/plot_results.py")
        print("Use this episodes CSV:", episodes_csv)

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
        "notebooks/plot_comare.py",      # your renamed one
        "notebooks/compare_results.py",  # common alt
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
        "Train",
        "Watch (play current model)",
        "Plot single persona",
        "Compare personas",
        "Quit",
    ]
    while True:
        choice = pick(MENU, "Main Menu")
        if choice == "Train":
            action_train()
        elif choice.startswith("Watch"):
            action_watch()
        elif choice.startswith("Plot single"):
            action_plot_single()
        elif choice.startswith("Compare"):
            action_plot_compare()
        else:
            print("bye")
            break

if __name__ == "__main__":
    main()
