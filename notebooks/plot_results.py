import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def line_plot(df, col, out):
    if col not in df.columns: return
    plt.figure()
    plt.plot(df.index, df[col])
    plt.title(col)
    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def bar_totals(summary_csv, out, episodes_csv=None):
    cols, vals = [], []

    if summary_csv and os.path.exists(summary_csv):
        sdf = pd.read_csv(summary_csv)
        cols = [c for c in sdf.columns if c.startswith("total_")]
        if cols:
            vals = [float(sdf.loc[0, c]) for c in cols]

    if not cols and episodes_csv and os.path.exists(episodes_csv):
        df = pd.read_csv(episodes_csv)
        if len(df) > 0:
            last = df.iloc[-1]
            cols = [c for c in df.columns if c.startswith("total_")]
            vals = [float(last[c]) for c in cols] if cols else []

    if not cols:
        return

    plt.figure()
    plt.bar(cols, vals)
    plt.title("Totals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def hist_plot(df, col, out):
    if col not in df.columns: return
    plt.figure()
    df[col].plot(kind="hist", bins=20)
    plt.title(f"{col} distribution")
    plt.xlabel(col); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_csv", required=True, help="per-episode CSV (eval OR training)")
    ap.add_argument("--summary_csv", default=None, help="eval summary CSV with totals (optional)")
    ap.add_argument("--outdir", default=None, help="where to save plots (default: alongside CSV)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.episodes_csv)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.episodes_csv)

    # 1) Line plots (episode curves)
    line_plot(df, "ep_return", os.path.join(outdir, "ep_return.png"))
    line_plot(df, "ep_len",    os.path.join(outdir, "ep_len.png"))

    # 2) Histograms for common metrics if present
    for m in ["apples_eaten", "pellets", "deaths", "unique_tiles"]:
        hist_plot(df, m, os.path.join(outdir, f"{m}_hist.png"))

    # 3) Totals bar chart (eval summary OR last row of training CSV)
    if args.summary_csv is None:
        base = os.path.splitext(args.episodes_csv)[0]
        args.summary_csv = base + "_summary.csv"
    bar_totals(args.summary_csv, os.path.join(outdir, "totals.png"), episodes_csv=args.episodes_csv)

if __name__ == "__main__":
    main()
