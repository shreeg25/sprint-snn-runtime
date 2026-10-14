"""
baselines_patient_split.py -- replaces run_master_benchmarks.py for the paper.

Same patient-wise stratified split, AAMI labels, paced-record exclusion and
raw-ECG windows as train_halting_snn.py. Models are trained on TRAIN patients,
the decision threshold is chosen on VAL, and metrics are reported on TEST only.
(run_master_benchmarks.py trained and scored on the same balanced loader with
the old N-vs-{V,A,L,R} labels, so its numbers are training-set scores.)

Also prints multiply-accumulates (MACs) per beat so compute can be compared
against the halting model honestly.

Usage:
  python baselines_patient_split.py --data_dir data/raw --metrics_json path/to/metrics.json
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset

import train_halting_snn as ths
from eval_sprint import load_cfg, materialize, metrics_at, pick_threshold


class MLP(nn.Module):
    def __init__(self, t=100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(t, 128), nn.ReLU(), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


class CNN1D(nn.Module):
    def __init__(self, t=100):
        super().__init__()
        self.conv = nn.Conv1d(1, 16, kernel_size=5, stride=2)
        self.l_out = (t - 5) // 2 + 1
        self.fc = nn.Linear(16 * self.l_out, 2)

    def forward(self, x):
        return self.fc(torch.relu(self.conv(x.unsqueeze(1))).flatten(1))


def macs_mlp(t):
    return t * 128 + 128 * 2


def macs_cnn(t):
    l_out = (t - 5) // 2 + 1
    return 16 * 5 * l_out + 16 * l_out * 2


def macs_halting_per_step(cfg, in_features=3):
    h = cfg.hidden
    return in_features * h + h * h + h * 2 + h  # encoder + recurrent + class head + halt head


def train_torch(model, Xtr, ytr, class_w, device, epochs, lr=1e-3, bs=256, seed=42):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(weight=class_w.to(device))
    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for x, y in dl:
            opt.zero_grad()
            lossf(model(x.to(device)), y.to(device)).backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def scores(model, X, device):
    return torch.softmax(model(X.to(device)), -1)[:, 1].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/raw")
    ap.add_argument("--metrics_json", default=None,
                    help="metrics.json of the SPRINT run, so the split seed/T_max match.")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="baselines_patient_split.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = load_cfg(args.metrics_json, args.device)
    tr_ids, va_ids, te_ids = ths.stratified_patient_split(args.data_dir, seed=cfg.seed)
    data = {}
    for name, ids in [("train", tr_ids), ("val", va_ids), ("test", te_ids)]:
        X, y = materialize(ths.MITBIHBeatDataset(args.data_dir, ids, t_max=cfg.t_max))
        data[name] = (X.squeeze(-1), y)

    Xtr, ytr = data["train"]
    class_w = ths.compute_class_weights(ytr.numpy(), 2, pos_weight_mult=cfg.pos_weight_mult)

    results = []

    def report(name, s_val, s_test, macs, params):
        thr = pick_threshold(s_val, data["val"][1].numpy())
        m = metrics_at(s_test, data["test"][1].numpy(), thr)
        results.append({"model": name, "threshold_from_val": thr, "macs_per_beat": macs,
                        "params": params, **m})

    mlp = train_torch(MLP(cfg.t_max), Xtr, ytr, class_w, args.device, args.epochs, seed=cfg.seed)
    report("MLP (float32)", scores(mlp, data["val"][0], args.device),
           scores(mlp, data["test"][0], args.device), macs_mlp(cfg.t_max),
           sum(p.numel() for p in mlp.parameters()))

    cnn = train_torch(CNN1D(cfg.t_max), Xtr, ytr, class_w, args.device, args.epochs, seed=cfg.seed)
    report("1D-CNN (float32)", scores(cnn, data["val"][0], args.device),
           scores(cnn, data["test"][0], args.device), macs_cnn(cfg.t_max),
           sum(p.numel() for p in cnn.parameters()))

    rf = RandomForestClassifier(n_estimators=50, max_depth=10, class_weight="balanced",
                                random_state=cfg.seed, n_jobs=-1)
    rf.fit(Xtr.numpy(), ytr.numpy())
    report("Random Forest (50x depth-10)", rf.predict_proba(data["val"][0].numpy())[:, 1],
           rf.predict_proba(data["test"][0].numpy())[:, 1], None, None)

    per_step = macs_halting_per_step(cfg)
    print(f"\nHalting model compute: {per_step} MACs per timestep "
          f"-> {per_step * cfg.t_max} MACs for a full T={cfg.t_max} rollout "
          f"(multiply by mean latency from eval_sprint.py for early exit).\n")

    hdr = f"{'model':<30} {'P':>6} {'R':>6} {'F1':>6} {'BalAcc':>7} {'AUROC':>6} {'AUPRC':>6} {'MACs':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        macs = f"{r['macs_per_beat']:>8d}" if r["macs_per_beat"] else f"{'-':>8}"
        print(f"{r['model']:<30} {r['precision']*100:6.1f} {r['recall']*100:6.1f} {r['f1']*100:6.1f} "
              f"{r['balanced_acc']*100:7.1f} {r['auroc']*100:6.1f} {r['auprc']*100:6.1f} {macs}")

    with open(args.out, "w") as f:
        json.dump({"halting_macs_per_step": per_step, "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
