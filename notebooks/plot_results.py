import argparse, os, glob, sys
import pandas as pd
import matplotlib.pyplot as plt

def pick_csv_interactive():
    candidates = sorted(glob.glob("logs/*/*/train/episodes.csv"))
    if not candidates:
        print("No training CSVs found under logs/*/*/train/episodes.csv")
        sys.exit(1)
    print("\nSelect a run to plot:")
    for i, p in enumerate(candidates, 1):
        print(f"  [{i}] {p}")
    while True:
        try:
            n = int(input("Enter number: ").strip())
            if 1 <= n <= len(candidates):
                return candidates[n-1]
        except ValueError:
            pass

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
    import pandas as pd, os
    cols, vals = [], []

    # 1) Prefer summary CSV with total_* columns
    if summary_csv and os.path.exists(summary_csv):
        try:
            sdf = pd.read_csv(summary_csv)
            tcols = [c for c in sdf.columns if c.startswith("total_")]
            if len(sdf) > 0 and tcols:
                cols = tcols
                vals = [float(sdf.loc[0, c]) for c in tcols]
        except Exception:
            pass

    # 2) Fallback: use episodes.csv
    if (not cols) and episodes_csv and os.path.exists(episodes_csv):
        try:
            df = pd.read_csv(episodes_csv)
            # Try total_* on the last row first
            tcols = [c for c in df.columns if c.startswith("total_")]
            if len(df) > 0 and tcols:
                last = df.iloc[-1]
                cols = tcols
                vals = [float(last[c]) for c in tcols]
            else:
                # FINAL fallback: sum known per-episode counters if present
                candidates = ["apples_eaten", "pellets", "deaths", "unique_tiles",
                              "dots", "ghosts_eaten", "lives_lost"]
                present = [c for c in candidates if c in df.columns]
                if present:
                    cols = [f"total_{c}" for c in present]
                    vals = [float(df[c].sum()) for c in present]
        except Exception:
            pass

    if not cols:
        print("No totals available: no total_* columns and no known per-episode counters in CSV.")
        return

    import matplotlib.pyplot as plt
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
    ap.add_argument("--episodes_csv", default=None, help="per-episode CSV (eval OR training)")
    ap.add_argument("--summary_csv", default=None, help="eval summary CSV with totals (optional)")
    ap.add_argument("--outdir", default=None, help="output dir (default: alongside CSV)")
    args = ap.parse_args()

    if not args.episodes_csv:
        args.episodes_csv = pick_csv_interactive()

    outdir = args.outdir or os.path.dirname(args.episodes_csv)
    os.makedirs(outdir, exist_ok=True)

    df = pd.read_csv(args.episodes_csv)

    line_plot(df, "ep_return", os.path.join(outdir, "ep_return.png"))
    line_plot(df, "ep_len",    os.path.join(outdir, "ep_len.png"))

    for m in ["apples_eaten", "pellets", "deaths", "unique_tiles"]:
        hist_plot(df, m, os.path.join(outdir, f"{m}_hist.png"))

    if args.summary_csv is None:
        base = os.path.splitext(args.episodes_csv)[0]
        args.summary_csv = base + "_summary.csv"
    bar_totals(args.summary_csv, os.path.join(outdir, "totals.png"), episodes_csv=args.episodes_csv)

if __name__ == "__main__":
    main()
