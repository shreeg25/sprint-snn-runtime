"""
eval_sprint.py -- corrected evaluation for a trained SPRINT checkpoint.

Fixes relative to the evaluation inside train_halting_snn.py:
  * Decision thresholds are chosen on VALIDATION, then applied unchanged to test.
  * Every rollout runs the full T_max (early_exit=False). The old evaluate()
    broke out of the loop once a whole batch had hard-halted, leaving zero rows
    in mem_trains that the residual survival mass was then folded onto.
  * "SPRINT hard-exit" reports what a device that stops at the halt would see:
    the class readout at the FIRST hard halt spike, and that timestep as latency.
  * Same-checkpoint baselines on a no-halt rollout of the same weights:
      - fixed truncation at T=k (readout at step k; also mean readout to k)
      - confidence-only exit (first t >= t_min with max softmax prob >= c)

Usage:
  python eval_sprint.py --checkpoint path/to/best_model.pt \
      --metrics_json path/to/metrics.json --data_dir data/raw --out_dir eval_out
"""

import argparse
import csv
import dataclasses
import json
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

import train_halting_snn as ths

TRUNC_STEPS = (10, 20, 30, 40, 60, 100)
CONF_LEVELS = (0.60, 0.70, 0.80, 0.90, 0.95, 0.99)
THRESH_GRID = np.round(np.arange(0.02, 0.99, 0.02), 2)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

def load_cfg(metrics_json, device):
    """Rebuild the exact HaltConfig of the training run from metrics.json."""
    cfg = ths.HaltConfig()
    if metrics_json:
        with open(metrics_json) as f:
            saved = json.load(f).get("config", {})
        names = {fld.name for fld in dataclasses.fields(ths.HaltConfig)}
        for k, v in saved.items():
            if k in names:
                setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    else:
        print("[EVAL] WARNING: no --metrics_json given, using HaltConfig defaults. "
              "These may NOT match the run that produced the checkpoint.")
    cfg.device = device
    return cfg


def materialize(ds, bs=1024):
    """Draw every sample once so the halting and no-halt rollouts see the
    identical V_dd trace (MITBIHVoltageDataset draws alpha/noise statefully)."""
    xs, ys = [], []
    for x, y in DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0):
        xs.append(x)
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


# --------------------------------------------------------------------------
# Rollouts and readouts
# --------------------------------------------------------------------------

@torch.no_grad()
def rollout(model, X, device, bs=512, allow_halt=True):
    """Full-length rollout. Returns class_mem (T,N,2), hard halt spikes (T,N),
    halt_probs (T,N). allow_halt=False disables the learned halt and the
    emergency floor, so the same weights run as a plain full-sequence model."""
    saved = (model.lambda_base, model.v_emergency)
    if not allow_halt:
        model.lambda_base, model.v_emergency = 1e9, -1.0
    cm, sp, hp, hs = [], [], [], []
    try:
        for i in range(0, len(X), bs):
            spk, mem, hprob = model(X[i:i + bs].to(device), early_exit=False, voltage_gate=1.0)
            cm.append(mem[:, :, :2].cpu())
            sp.append(spk[:, :, 2].cpu())
            hp.append(hprob.cpu())
            hidden_spikes = getattr(model, "last_spikes", None)  # spiking model only
            if hidden_spikes is not None:
                hs.append(hidden_spikes.cpu())
    finally:
        model.lambda_base, model.v_emergency = saved
    return (torch.cat(cm, 1), torch.cat(sp, 1), torch.cat(hp, 1),
            torch.cat(hs, 1) if hs else None)


def compute_cost(latency, hidden_spikes, cfg, spiking):
    """Mean SOPs and MACs per beat up to the exit timestep.
    Spiking model: each hidden spike fans out to hidden (recurrent) + 2 (class)
    + 1 (halt) synapses = SOPs; the analog ECG encoder (1 -> hidden) and the two
    voltage inputs of the halt head stay MACs. Non-spiking model: everything is
    a MAC (3*h + h*h + 2*h + h per step)."""
    h = cfg.hidden
    if not spiking:
        return float("nan"), float((latency * (3 * h + h * h + 2 * h + h)).mean())
    steps = latency.long().clamp(min=1) - 1
    cum = hidden_spikes.cumsum(0)                      # (T, N)
    spikes_to_exit = cum[steps, torch.arange(cum.shape[1])]
    return float((spikes_to_exit * (h + 3)).mean()), float((latency * (h + 2)).mean())


def first_event_readout(class_mem, event):
    """Readout at the first True in event (T,N); falls back to the last step."""
    T, N = event.shape
    fired = event.any(0)
    idx = torch.where(fired, event.float().argmax(0), torch.full((N,), T - 1, dtype=torch.long))
    return class_mem[idx, torch.arange(N)], (idx + 1).float()


def soft_readout(class_mem, halt_probs, t_min):
    """The paper's p_stop-weighted readout, computed on a full rollout."""
    p = halt_probs.clone()
    if t_min > 1:
        p[: t_min - 1] = 0.0
    cont = 1.0 - p
    S = torch.cumprod(torch.cat([torch.ones_like(cont[:1]), cont[:-1]]), 0)
    p_stop = S * p
    p_stop[-1] = p_stop[-1] + S[-1] * (1.0 - p[-1])
    return (p_stop.unsqueeze(-1) * class_mem).sum(0), S.sum(0)


def all_methods(halt_run, plain_run, t_min):
    """Returns {method_name: (logits (N,2), latency (N,), from_halt_run: bool)}."""
    out = _methods(halt_run, plain_run, t_min)
    return {k: (lg, lat, k.startswith("SPRINT")) for k, (lg, lat) in out.items()}


def _methods(halt_run, plain_run, t_min):
    cm_h, sp_h, hp_h, _ = halt_run
    cm_p = plain_run[0]
    T, N, _ = cm_p.shape
    out = {}
    out["SPRINT hard-exit"] = first_event_readout(cm_h, sp_h > 0.5)
    out["SPRINT soft readout (paper)"] = soft_readout(cm_h, hp_h, t_min)
    for k in TRUNC_STEPS:
        k = min(k, T)
        lat = torch.full((N,), float(k))
        out[f"Truncate T={k} (last)"] = (cm_p[k - 1], lat)
        out[f"Truncate T={k} (mean)"] = (cm_p[:k].mean(0), lat)
    conf = torch.softmax(cm_p, -1).max(-1).values  # (T,N)
    for c in CONF_LEVELS:
        ev = conf >= c
        ev[: max(t_min - 1, 0)] = False
        out[f"Confidence exit c={c:.2f}"] = first_event_readout(cm_p, ev)
    return out


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def pos_score(logits):
    return torch.softmax(logits.float(), -1)[:, 1].numpy()


def metrics_at(score, y, thr):
    pred = score > thr
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    both = len(np.unique(y)) == 2
    return {
        "precision": p, "recall": r, "f1": f1, "specificity": spec,
        "balanced_acc": (r + spec) / 2,
        "auroc": float(roc_auc_score(y, score)) if both else float("nan"),
        "auprc": float(average_precision_score(y, score)) if both else float("nan"),
    }


def pick_threshold(score, y):
    best_t, best_f1 = 0.5, -1.0
    for t in THRESH_GRID:
        f1 = metrics_at(score, y, t)["f1"]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--metrics_json", default=None)
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--out_dir", default="eval_out")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_module", default=None,
                    help="Optional module providing build_model(in_features, cfg) "
                         "(e.g. train_halting_lif for the retrained spiking model).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = load_cfg(args.metrics_json, args.device)
    print(f"[EVAL] t_min={cfg.t_min} train_alpha_range={cfg.train_alpha_range} "
          f"alpha_grid={cfg.eval_alpha_grid} seed={cfg.seed}")

    _, val_loader, _, _, test_base = ths.load_mit_bih_data(
        args.data_dir, t_max=cfg.t_max, batch_size=cfg.batch_size, seed=cfg.seed,
        alpha_range=tuple(cfg.train_alpha_range),
    )

    build = ths.build_model
    if args.model_module:
        build = __import__(args.model_module).build_model
    model = build(3, cfg).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    # Thresholds: chosen once per method on validation (random alpha, as in training).
    Xv, yv = materialize(val_loader.dataset)
    val_methods = all_methods(rollout(model, Xv, args.device),
                              rollout(model, Xv, args.device, allow_halt=False), cfg.t_min)
    yv_np = yv.numpy()
    thresholds = {m: pick_threshold(pos_score(lg), yv_np) for m, (lg, _, _) in val_methods.items()}

    # Test: random-alpha test set (same distribution as training) + fixed-alpha grid.
    test_sets = [("random", ths.MITBIHVoltageDataset(
        test_base, alpha_range=tuple(cfg.train_alpha_range), seed=cfg.seed + 2))]
    test_sets += [(a, ths.MITBIHVoltageDataset(test_base, fixed_alpha=a, noise_std=0.01,
                                                seed=cfg.seed + 100))
                  for a in cfg.eval_alpha_grid]

    rows = []
    for alpha, ds in test_sets:
        Xt, yt = materialize(ds)
        yt_np = yt.numpy()
        halt_run = rollout(model, Xt, args.device)
        plain_run = rollout(model, Xt, args.device, allow_halt=False)
        spiking = halt_run[3] is not None
        methods = all_methods(halt_run, plain_run, cfg.t_min)
        for name, (lg, lat, from_halt) in methods.items():
            m = metrics_at(pos_score(lg), yt_np, thresholds[name])
            sops, macs = compute_cost(lat, (halt_run if from_halt else plain_run)[3], cfg, spiking)
            rows.append({"alpha": alpha, "method": name, "mean_latency": float(lat.mean()),
                         "mean_sops": sops, "mean_macs": macs,
                         "threshold_from_val": thresholds[name], **m})

    keys = list(rows[0].keys())
    with open(os.path.join(args.out_dir, "eval_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump({"checkpoint": args.checkpoint, "rows": rows}, f, indent=2)

    hdr = (f"{'alpha':>7} {'method':<30} {'T':>6} {'P':>6} {'R':>6} {'F1':>6} {'BalAcc':>7} "
           f"{'AUROC':>6} {'AUPRC':>6} {'SOPs':>9} {'MACs':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        a = r["alpha"] if isinstance(r["alpha"], str) else f"{r['alpha']:.3f}"
        print(f"{a:>7} {r['method']:<30} {r['mean_latency']:6.1f} {r['precision']*100:6.1f} "
              f"{r['recall']*100:6.1f} {r['f1']*100:6.1f} {r['balanced_acc']*100:7.1f} "
              f"{r['auroc']*100:6.1f} {r['auprc']*100:6.1f} {r['mean_sops']:9.0f} {r['mean_macs']:9.0f}")
    print(f"[EVAL] Wrote {args.out_dir}/eval_results.csv and .json")


if __name__ == "__main__":
    main()
