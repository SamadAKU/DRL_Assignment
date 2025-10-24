import argparse, os, glob
import pandas as pd
import matplotlib.pyplot as plt

def load_runs(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    runs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            for col in ["app","algo","persona","run_id"]:
                if col not in df.columns:
                    raise ValueError(f"{f} missing '{col}'")
            runs.append((f, df))
        except Exception as e:
            print("skip", f, e)
    return runs

def plot_compare(runs, metric, out_png):
    if not runs: return
    plt.figure()
    for f, df in runs:
        label = f"{df['app'].iloc[0]}/{df['algo'].iloc[0]}/{df['persona'].iloc[0]} ({df['run_id'].iloc[0]})"
        if metric in df.columns:
            plt.plot(df.index, df[metric], label=label)
    plt.title(metric)
    plt.xlabel("Episode")
    plt.ylabel(metric)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print("saved", out_png)

def build_summary(runs, out_csv):
    rows = []
    for f, df in runs:
        meta = {k: df[k].iloc[0] for k in ["app","algo","persona","run_id"]}
        last_ts = int(df["total_timesteps"].iloc[-1]) if "total_timesteps" in df.columns else None
        row = {
            **meta,
            "episodes": len(df),
            "final_ep_return": float(df["ep_return"].iloc[-1]) if "ep_return" in df.columns else None,
            "median_ep_return": float(df["ep_return"].median()) if "ep_return" in df.columns else None,
            "final_ep_len": float(df["ep_len"].iloc[-1]) if "ep_len" in df.columns else None,
            "median_ep_len": float(df["ep_len"].median()) if "ep_len" in df.columns else None,
            "last_total_timesteps": last_ts,
            "source_csv": f,
        }
        for c in df.columns:
            if c.startswith("total_"):
                row[f"last_{c}"] = float(df[c].iloc[-1])
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("saved", out_csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="CSV files or globs (e.g. logs/*/train/episodes.csv)")
    ap.add_argument("--outdir", default="compare_out", help="output dir for plots and summary")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    runs = load_runs(args.inputs)

    plot_compare(runs, "ep_return", os.path.join(args.outdir, "compare_ep_return.png"))
    plot_compare(runs, "ep_len",    os.path.join(args.outdir, "compare_ep_len.png"))
    build_summary(runs, os.path.join(args.outdir, "summary_by_run.csv"))

if __name__ == "__main__":
    main()
