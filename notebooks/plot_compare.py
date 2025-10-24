import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", required=True, choices=["snake","pacman"])
    ap.add_argument("--personas", nargs="+", required=True, help="e.g. pacman_explorer pacman_survivor")
    ap.add_argument("--metric", default="ep_return", help="column to compare (default: ep_return)")
    ap.add_argument("--out", default=None, help="output png path")
    args = ap.parse_args()

    dfs = {}
    for persona in args.personas:
        path = f"logs/{args.app}/{persona}/train/episodes.csv"
        if os.path.exists(path):
            df = pd.read_csv(path)
            if args.metric in df.columns:
                dfs[persona] = df
        else:
            print(f"missing: {path}")

    if not dfs:
        print("No data found.")
        return

    plt.figure()
    for persona, df in dfs.items():
        plt.plot(df.index, df[args.metric], label=persona)
    plt.title(f"{args.app} — {args.metric} vs episode")
    plt.xlabel("Episode"); plt.ylabel(args.metric)
    plt.legend()
    plt.tight_layout()

    out = args.out or f"logs/{args.app}/compare_{args.metric}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out); plt.close()
    print("saved", out)

if __name__ == "__main__":
    main()
