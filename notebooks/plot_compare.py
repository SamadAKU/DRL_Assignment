#!/usr/bin/env python3
import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

# Compare one metric across multiple personas.
# Expects episodes.csv at logs/{app}/{persona}/episodes.csv unless --episodes_glob is provided.

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["snake","pacman"])
    ap.add_argument("--personas", nargs="+", required=True, help="e.g. pacman_explorer pacman_survivor")
    ap.add_argument("--metric", default="ep_return", help="column to compare (default: ep_return)")
    ap.add_argument("--episodes_glob", default=None, help='Override pattern like "logs/*/episodes.csv"')
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    dfs = {}
    for persona in args.personas:
        if args.episodes_glob:
            matches = glob.glob(args.episodes_glob.format(app=args.app, persona=persona))
        else:
            path = f"logs/{args.app}/{persona}/episodes.csv"
            matches = [path] if os.path.exists(path) else []
        if not matches:
            print("warn: no episodes.csv for", persona)
            continue
        df = pd.read_csv(matches[0]).reset_index(drop=True)
        if args.metric not in df.columns:
            have = ", ".join([c for c in df.columns if c.startswith("ep_") or "eaten" in c or "return" in c or "len" in c])
            print(f"warn: {args.metric} not in columns for {persona}. have: {have}")
            continue
        dfs[persona] = df

    if not dfs:
        raise SystemExit("Nothing to plot. Check your paths/metric.")

    plt.figure()
    for persona, df in dfs.items():
        plt.plot(df.index, df[args.metric], label=persona)
    plt.title(f"{args.app} — {args.metric} vs episode")
    plt.xlabel("Episode"); plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout()

    out = args.out or f"logs/{args.app}/compare_{args.metric}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", out)

if __name__ == "__main__":
    main()
