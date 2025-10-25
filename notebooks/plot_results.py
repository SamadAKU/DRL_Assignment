#!/usr/bin/env python3
import argparse, os, re
import pandas as pd
import matplotlib.pyplot as plt

# No seaborn. Single-figure per plot. No explicit colors.

DEFAULT_COLS = [
    "ep_return","ep_len",
    "pellets_eaten","apples_eaten",
    "food_eaten","dots_eaten","fruits_eaten",
]

CUSTOM_MATCH = re.compile(r"(apple|pellet|food|dot|cookie|fruit)", re.I)

def line_plot(df, col, out):
    if col not in df.columns:
        return False
    plt.figure()
    plt.plot(df.index, df[col])
    plt.title(col)
    plt.xlabel("Episode")
    plt.ylabel(col)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return True

def compute_totals_from_episodes(df):
    totals = {}
    for c in df.columns:
        if CUSTOM_MATCH.search(c):
            try:
                totals[c] = float(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
            except Exception:
                pass
    return pd.DataFrame({"metric": list(totals.keys()), "total": list(totals.values())})

def bar_totals(sdf, out):
    if sdf is None or sdf.empty: 
        return False
    sdf = sdf.sort_values("total", ascending=False)
    plt.figure()
    plt.bar(sdf["metric"], sdf["total"])
    plt.xticks(rotation=30, ha="right")
    plt.title("Totals across episodes")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    return True

def autodetect_metric_cols(df):
    cols = []
    for c in DEFAULT_COLS:
        if c in df.columns:
            cols.append(c)
    # include any other custom-matching columns that exist
    for c in df.columns:
        if CUSTOM_MATCH.search(c) and c not in cols:
            cols.append(c)
    # Always keep unique order
    seen = set()
    uniq = []
    for c in cols:
        if c not in seen:
            uniq.append(c); seen.add(c)
    return uniq

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes_csv", required=True, help="Path to episodes.csv (per-episode metrics)")
    ap.add_argument("--outdir", default=None, help="Folder to save plots (default: alongside episodes.csv in plots/)")
    ap.add_argument("--lines", nargs="*", default=None, help="Columns to plot as line charts; if omitted, autodetect common/custom metrics")
    ap.add_argument("--summary_csv", default=None, help="Optional precomputed totals CSV; if absent, derived from episodes")
    args = ap.parse_args()

    if not os.path.exists(args.episodes_csv):
        raise SystemExit(f"episodes file not found: {args.episodes_csv}")

    df = pd.read_csv(args.episodes_csv)
    df = df.reset_index(drop=True)  # episode index

    outdir = args.outdir or os.path.join(os.path.dirname(args.episodes_csv), "plots")
    os.makedirs(outdir, exist_ok=True)

    # 1) Line plots
    cols = args.lines if args.lines else autodetect_metric_cols(df)
    generated = []
    for col in cols:
        out = os.path.join(outdir, f"{col}.png")
        if line_plot(df, col, out):
            print("saved", out)
            generated.append(out)

    # 2) Totals bar chart of custom metrics
    if args.summary_csv and os.path.exists(args.summary_csv):
        sdf = pd.read_csv(args.summary_csv)
    else:
        sdf = compute_totals_from_episodes(df)
        if not sdf.empty:
            derived_path = os.path.join(outdir, "totals_summary.csv")
            sdf.to_csv(derived_path, index=False)
            print("saved", derived_path)

    if not sdf.empty:
        bar_path = os.path.join(outdir, "totals.png")
        if bar_totals(sdf, bar_path):
            print("saved", bar_path)

if __name__ == "__main__":
    main()
