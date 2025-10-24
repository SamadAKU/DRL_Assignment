import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", required=True, help="logs/{app}/train")
    p.add_argument("--out", default=None, help="output png path")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.log_dir, "monitor*.csv")))
    if not files:
        # fallback: single monitor.csv
        f = os.path.join(args.log_dir, "monitor.csv")
        if os.path.exists(f): files = [f]
    if not files:
        print("No monitor.csv files found in", args.log_dir)
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, comment='#')
            if "r" in df.columns and "l" in df.columns and "t" in df.columns:
                df["source"] = os.path.basename(f)
                dfs.append(df)
        except Exception as e:
            print("skip", f, e)

    if not dfs:
        print("No valid monitor.csv data")
        return

    all_df = pd.concat(dfs, ignore_index=True).sort_values("t")
    all_df["ewm_r"] = all_df["r"].ewm(alpha=0.02).mean()

    plt.figure()
    plt.plot(all_df["t"], all_df["r"], alpha=0.3, label="episode reward")
    plt.plot(all_df["t"], all_df["ewm_r"], label="EWMA reward")
    plt.xlabel("time (seconds)")
    plt.ylabel("episode reward")
    plt.title(f"Training reward – {args.log_dir}")
    plt.legend()
    plt.tight_layout()
    out = args.out or os.path.join(args.log_dir, "training_reward.png")
    plt.savefig(out)
    print("Saved", out)

if __name__ == "__main__":
    main()
