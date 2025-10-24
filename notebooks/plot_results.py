import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

def line_plot(df, col, out):
    if col not in df.columns: 
        return
    plt.figure()
    plt.plot(df.index, df[col])
    plt.title(col)
    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def hist_plot(df, col, out):
    if col not in df.columns: 
        return
    plt.figure()
    df[col].plot(kind="hist", bins=20)
    plt.title(f"{col} distribution")
    plt.xlabel(col); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def compute_totals_from_episodes(df):
    # Sum any numeric metrics that look like counts/aggregates we logged
    candidate_cols = [c for c in df.columns if c not in ["ep_len","ep_return"]]
    num_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    # Also include ep_len and ep_return totals for completeness
    if "ep_len" in df.columns: num_cols.append("ep_len")
    if "ep_return" in df.columns: num_cols.append("ep_return")
    totals = {f"total_{c}": float(df[c].sum()) for c in num_cols}
    return pd.DataFrame([totals])

def bar_totals(summary_df, out):
    cols = [c for c in summary_df.columns if c.startswith("total_")]
    if not cols: 
        return
    plt.figure()
    plt.bar(cols, [float(summary_df.loc[0, c]) for c in cols])
    plt.title("Totals")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out); plt.close()
    print("saved", out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_csv", required=True, help="per-episode CSV (training or eval)")
    ap.add_argument("--summary_csv", default=None, help="optional summary CSV with totals")
    ap.add_argument("--outdir", default=None, help="where to save plots (default: alongside CSV)")
    args = ap.parse_args()

    outdir = args.outdir or os.path.dirname(args.episodes_csv)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.episodes_csv)

    # 1) Line plots (episode curves)
    line_plot(df, "ep_return", os.path.join(outdir, "ep_return.png"))
    line_plot(df, "ep_len",    os.path.join(outdir, "ep_len.png"))

    # 2) Histograms for common metrics if present
    for m in ["apples_eaten", "pellets", "deaths", "unique_tiles", "score_delta"]:
        hist_plot(df, m, os.path.join(outdir, f"{m}_hist.png"))

    # 3) Totals bar chart
    if args.summary_csv and os.path.exists(args.summary_csv):
        sdf = pd.read_csv(args.summary_csv)
    else:
        sdf = compute_totals_from_episodes(df)
        # also save a derived summary next to episodes.csv
        base = os.path.splitext(args.episodes_csv)[0]
        derived_path = base + "_summary.csv"
        sdf.to_csv(derived_path, index=False)

    bar_totals(sdf, os.path.join(outdir, "totals.png"))

if __name__ == "__main__":
    main()
