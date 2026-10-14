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
import dataclasses
import glob
import json
import os

import torch

import train_halting_snn as ths

CURRENT_KEYS = {f.name for f in dataclasses.fields(ths.HaltConfig)}


def compatible(run, cfg_dict, build):
    """True only if the run was made by the CURRENT script version: its config has
    every field the current HaltConfig has, and its weights load strictly into
    the current model. Older runs used different code (and a different val
    metric path), so their val AUPRC is not comparable."""
    missing = CURRENT_KEYS - set(cfg_dict) - {"device"}
    if missing:
        return False, f"config missing {sorted(missing)[:3]}"
    cfg = ths.HaltConfig()
    for k, v in cfg_dict.items():
        if k in CURRENT_KEYS and k != "device":
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    cfg.device = "cpu"
    try:
        sd = torch.load(os.path.join(run, "best_model.pt"), map_location="cpu")
        build(3, cfg).load_state_dict(sd, strict=True)
    except Exception as e:
        return False, f"weights don't fit current model ({type(e).__name__})"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--since", default=None,
                    help="Only runs at or after this id, e.g. run_20260910_000000")
    ap.add_argument("--model_module", default=None,
                    help="train_halting_lif for the spiking retrain")
    args = ap.parse_args()
    build = __import__(args.model_module).build_model if args.model_module else ths.build_model
    skipped = []

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
        if args.since and os.path.basename(run) < args.since:
            continue
        eps = d.get("epochs", [])
        c = d.get("config", {})
        if not eps or not any("val_auprc" in e for e in eps):
            skipped.append((os.path.basename(run), "no val_auprc logged"))
            continue
        ok, why = compatible(run, c, build)
        if not ok:
            skipped.append((os.path.basename(run), why))
            continue
        best = max(eps, key=lambda e: e.get("val_auprc", -1))
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

    if skipped:
        print(f"Skipped {len(skipped)} incompatible/older runs:")
        for r, why in sorted(skipped):
            print(f"  {r}: {why}")
        print()
    if not rows:
        print("No compatible runs found.")
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
          f"--out_dir eval_{top['run']}"
          + (f" --model_module {args.model_module}" if args.model_module else ""))


if __name__ == "__main__":
    main()