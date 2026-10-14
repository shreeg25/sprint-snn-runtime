"""
rank_checkpoints.py -- choose a checkpoint by VALIDATION score only.

Each run's best_model.pt is already the epoch with the best val AUPRC
(see train_halting_snn.main). This ranks runs by that number. Evaluate only
the top run on test with eval_sprint.py; ranking runs by test score would be
tuning on the test set.

Usage:
  python rank_checkpoints.py --checkpoint_dir checkpoints
"""

import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    args = ap.parse_args()

    rows = []
    for mj in glob.glob(os.path.join(args.checkpoint_dir, "run_*", "metrics.json")):
        run = os.path.dirname(mj)
        if not os.path.exists(os.path.join(run, "best_model.pt")):
            continue
        try:
            with open(mj) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        eps = d.get("epochs", [])
        if not eps:
            continue
        best = max(eps, key=lambda e: e.get("val_auprc", -1))
        c = d.get("config", {})
        rows.append({
            "run": os.path.basename(run),
            "val_auprc": best.get("val_auprc", float("nan")),
            "epoch": best["epoch"] + 1,
            "n_epochs": len(eps),
            "finished": "power_latency_tradeoff" in d,
            "seed": c.get("seed"),
            "lr": c.get("lr"),
            "pos_w": c.get("pos_weight_mult"),
            "alpha": tuple(c.get("train_alpha_range", ())),
        })

    if not rows:
        print("No runs with both metrics.json and best_model.pt found.")
        return

    rows.sort(key=lambda r: r["val_auprc"], reverse=True)
    seeds = {r["seed"] for r in rows}
    print(f"{'run':<22} {'val_AUPRC':>9} {'ep':>6} {'done':>5} {'seed':>5} {'lr':>8} {'pos_w':>6}  alpha_range")
    for r in rows:
        print(f"{r['run']:<22} {r['val_auprc']:9.4f} {r['epoch']:>2}/{r['n_epochs']:<3} "
              f"{'yes' if r['finished'] else 'NO':>5} {str(r['seed']):>5} {str(r['lr']):>8} "
              f"{str(r['pos_w']):>6}  {r['alpha']}")
    if len(seeds) > 1:
        print(f"\nNOTE: runs use different seeds {sorted(seeds, key=str)}, so their patient splits "
              f"differ and val AUPRC is not directly comparable across seeds. "
              f"Compare within one seed (the paper's split is seed 42).")
    top = rows[0]
    print(f"\nTop by validation: {top['run']}")
    print(f"  python eval_sprint.py --checkpoint {args.checkpoint_dir}/{top['run']}/best_model.pt "
          f"--metrics_json {args.checkpoint_dir}/{top['run']}/metrics.json --data_dir data/raw "
          f"--out_dir eval_{top['run']}")


if __name__ == "__main__":
    main()
