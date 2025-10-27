import os, glob, math
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
LOG_ROOT = Path("logs")  
PERSONAS = {
    "pacman":   {"survivor": "pacman_survivor",  "explorer": "pacman_explorer"},
    "snake":    {"survivor": "snake_survivor",   "explorer": "snake_explorer"},
}
ALGOS = ["ppo", "a2c"]
ROLL = 100  
SAVE_DIR = Path("logs/plots")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def find_episodes_csv(app: str, persona: str, algo: str) -> str:
    """
    Looks for logs/<app>/<persona>/<algo>/episodes_train.csv
    Falls back to any episodes_train.csv under that tree.
    """
    root = LOG_ROOT / app / persona / algo
    exact = root / "episodes_train.csv"
    if exact.exists():
        return str(exact)
    matches = list(root.rglob("episodes_train.csv"))
    return str(matches[0]) if matches else ""

def load_run(app: str, persona: str, algo: str) -> pd.DataFrame:
    path = find_episodes_csv(app, persona, algo)
    if not path:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "ep_return" not in df.columns:
        candidates = [c for c in df.columns if "return" in c.lower() or "rew" in c.lower()]
        if not candidates:
            return pd.DataFrame()
        df = df.rename(columns={candidates[0]: "ep_return"})
    if "ep_len" not in df.columns:
        candidates = [c for c in df.columns if "len" in c.lower()]
        if candidates:
            df = df.rename(columns={candidates[0]: "ep_len"})
    if "timestamp_ms" not in df.columns:
        df["timestamp_ms"] = np.nan

    if df["timestamp_ms"].notna().sum() > 10:
        t0 = df["timestamp_ms"].dropna().iloc[0]
        df["time_hours"] = (df["timestamp_ms"].fillna(method="ffill") - t0) / (1000*60*60)
        df["x"] = df["time_hours"]
        x_label = "Time (hours)"
    else:
        df["x"] = np.arange(len(df), dtype=float)
        x_label = "Episode"

    # Rolling means
    df["ep_ret_roll"] = df["ep_return"].rolling(ROLL, min_periods=max(5, ROLL//5)).mean()
    if "ep_len" in df.columns:
        df["ep_len_roll"] = df["ep_len"].rolling(ROLL, min_periods=max(5, ROLL//5)).mean()

    df.attrs["x_label"] = x_label
    df.attrs["path"] = path
    return df

def label_for(algo: str, mode: str) -> str:
    return f"{algo.upper()} – {mode.capitalize()}"

def plot_algo_compare_for_game_mode(app: str, mode: str):
    """
    One plot comparing PPO vs A2C for a given game (app) and mode (survivor/explorer).
    """
    persona = PERSONAS[app][mode]
    dfs = {}
    for algo in ALGOS:
        df = load_run(app, persona, algo)
        if not df.empty:
            dfs[algo] = df
    if not dfs:
        print(f"[skip] No data for {app}-{mode}")
        return

    x_label = next(iter(dfs.values())).attrs.get("x_label", "Episode")
    plt.figure(figsize=(10,6))
    for algo, df in dfs.items():
        plt.plot(df["x"], df["ep_ret_roll"], label=f"{algo.upper()}")

    plt.title(f"{app.capitalize()} – {mode.capitalize()}: Reward (Rolling mean {ROLL})")
    plt.xlabel(x_label); plt.ylabel("Reward (rolling mean)")
    plt.legend()
    out = SAVE_DIR / f"{app}_{mode}_ppo_vs_a2c_reward.png"
    plt.tight_layout(); plt.savefig(out, dpi=140)
    plt.close()
    print("Wrote:", out)

def plot_game_overview(app: str):
    """
    One plot per game including all four lines: PPO/A2C × Survivor/Explorer
    """
    plt.figure(figsize=(11,7))
    x_label = "Episode"
    plotted = 0
    for mode in ["survivor", "explorer"]:
        persona = PERSONAS[app][mode]
        for algo in ALGOS:
            df = load_run(app, persona, algo)
            if df.empty:
                continue
            x_label = df.attrs.get("x_label", x_label)
            plt.plot(df["x"], df["ep_ret_roll"], label=label_for(algo, mode))
            plotted += 1
    if plotted == 0:
        print(f"[skip] No data for {app} overview")
        plt.close()
        return
    plt.title(f"{app.capitalize()} – Reward overview (Rolling mean {ROLL})")
    plt.xlabel(x_label); plt.ylabel("Reward (rolling mean)")
    plt.legend(ncol=2)
    out = SAVE_DIR / f"{app}_overview_reward.png"
    plt.tight_layout(); plt.savefig(out, dpi=140)
    plt.close()
    print("Wrote:", out)

def plot_extras(app: str):
    """
    Optional: extra graphs such as episode length trends and reward variance.
    Saves two plots if data available.
    """
    # Length trends (four lines if available)
    plt.figure(figsize=(11,7))
    plotted = 0; x_label = "Episode"
    for mode in ["survivor", "explorer"]:
        persona = PERSONAS[app][mode]
        for algo in ALGOS:
            df = load_run(app, persona, algo)
            if df.empty or "ep_len_roll" not in df.columns:
                continue
            x_label = df.attrs.get("x_label", x_label)
            plt.plot(df["x"], df["ep_len_roll"], label=f"{algo.upper()} – {mode}")
            plotted += 1
    if plotted:
        plt.title(f"{app.capitalize()} – Episode length (Rolling mean {ROLL})")
        plt.xlabel(x_label); plt.ylabel("Episode length (rolling mean)")
        plt.legend(ncol=2)
        out = SAVE_DIR / f"{app}_overview_length.png"
        plt.tight_layout(); plt.savefig(out, dpi=140)
        plt.close()
        print("Wrote:", out)
    else:
        plt.close()

    plt.figure(figsize=(11,7))
    plotted = 0; x_label = "Episode"
    for mode in ["survivor", "explorer"]:
        persona = PERSONAS[app][mode]
        for algo in ALGOS:
            df = load_run(app, persona, algo)
            if df.empty:
                continue
            x_label = df.attrs.get("x_label", x_label)
            roll_std = df["ep_return"].rolling(ROLL, min_periods=max(5, ROLL//5)).std()
            plt.plot(df["x"], roll_std, label=f"{algo.upper()} – {mode}")
            plotted += 1
    if plotted:
        plt.title(f"{app.capitalize()} – Reward rolling std (window={ROLL})")
        plt.xlabel(x_label); plt.ylabel("Reward rolling std")
        plt.legend(ncol=2)
        out = SAVE_DIR / f"{app}_overview_reward_std.png"
        plt.tight_layout(); plt.savefig(out, dpi=140)
        plt.close()
        print("Wrote:", out)
    else:
        plt.close()


def main():
    
    for app in ["pacman", "snake"]:
        for mode in ["survivor", "explorer"]:
            plot_algo_compare_for_game_mode(app, mode)

   
    for app in ["pacman", "snake"]:
        plot_game_overview(app)

    
    for app in ["pacman", "snake"]:
        plot_extras(app)

if __name__ == "__main__":
    main()
