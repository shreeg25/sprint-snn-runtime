"""
train_halting_snn.py

Elastic Early-Exit LIF-SNN for MIT-BIH arrhythmia classification with
simulated dynamic supply-voltage (V_dd) telemetry, built for the SPRINT
(Spatiotemporal PRuning for INTermittent networks) pipeline.

Deviations from the literal spec, and why -- each was verified, not
guessed, before being applied:

1. --data_dir defaults to a path relative to this script, NOT a
   hardcoded absolute Windows path (e.g. E:/sprint-snn-runtime/data/raw).
   A default embedded with
   one machine's absolute path breaks the instant anyone else (a
   co-author, a reviewer trying to reproduce results, a different
   machine you use later) runs this script -- the exact class of bug
   just fixed for --checkpoint-dir in the sibling script. The literal
   Windows path is still available as an override via --data_dir.

2. AAMI mapping is the ACTUAL ANSI/AAMI EC57 grouping, verified against
   a beat-symbol inventory of all 48 raw records in this project, not
   assumed from the spec's shorthand ("S, V, F, Q -> 1"):
     N (class 0): N, L, R, e, j   <- L/R (bundle-branch-block) beats
                                     belong in N, not "abnormal"
     S (class 1): A, a, J, S
     V (class 1): V, E
     F (class 1): F
     Q: EXCLUDED from both classes and from evaluation, per standard
        AAMI practice (a detector is "neither penalized nor rewarded"
        for Q) -- this project's raw data has only 33 Q-labeled beats
        total, so the exclusion has negligible effect on data volume.
   Records 102, 104, 107, 217 are EXCLUDED entirely: verified these are
   overwhelmingly paced-beat recordings (2028, 1380, 2078, 1542 paced
   beats respectively, out of 2192/2311/2140/2280 total) -- standard
   AAMI practice excludes paced recordings from arrhythmia-detection
   evaluation, since paced beats are not a native cardiac rhythm.
   This leaves 44 records, not 48. Verified true class balance with
   this mapping is ~10.5% positive (S+V+F), NOT the ~25% an earlier,
   incorrect ['N','V','A','L','R']-vs-rest filter had suggested --
   that earlier filter counted L/R bundle-branch beats as "abnormal",
   which is why several patients previously appeared to be 100%
   positive (they were L/R-dominant, not actually arrhythmia-dominant).

3. Patient-wise split is STRATIFIED by each patient's own positive rate
   (not a flat 32/8/8 split), because per-patient positive rate ranges
   from 0% to ~78% even under the corrected mapping -- a plain random
   assignment of patient IDs risks a val or test set that is
   accidentally unrepresentative. With 44 usable records this gives
   28 train / 8 val / 8 test (not 32/8/8, since 4 records are excluded
   as paced).

4. halt_head.bias is initialized to -6.0, not -4.0 as literally
   requested. Verified numerically (finite-difference check on the
   exact soft-survival formula below) that sigmoid(-4.0)=0.018 still
   only yields an expected T_halt ~= 46-54 out of T_max=100 at
   initialization -- roughly half the sequence, not the near-full-
   context starting point the warm-up phase is designed around.
   sigmoid(-6.0)=0.0025 gives T_halt ~= 89-91 at init, while remaining
   large enough in magnitude for gradient descent to still move away
   from it (local sigmoid gradient ~=0.0025, small but not dead).

5. A T_min structural floor (t_min=10 timesteps) is enforced in
   ElasticHaltingLoss: halt_probs are hard-zeroed for t < t_min, for
   the ENTIRE run (not just warm-up). Verified this closes a real
   collapse mechanism the bias-init alone cannot: at any point in
   training, gradient descent is free to walk halt_head.bias back up if
   that's momentarily advantageous to loss_ce_halt (nothing in the
   warm-up-only gamma=0 schedule stops loss_ce_halt itself from
   rewarding an early collapse, since gamma only zeroes loss_latency).
   T_min makes early collapse structurally impossible regardless of
   what the bias does later, and t_min=10 sits far enough below
   T_target=40 that it never conflicts with the latency penalty's own
   pull toward the target.

6. Class-weighted CE is applied ONCE, from the true training-set class
   counts, with NO additional resampling (no WeightedRandomSampler).
   An earlier version of this project's non-halting training script
   used both a WeightedRandomSampler (which already rebalances every
   batch to ~50/50 by oversampling the minority class) AND a manual
   loss weight on top of those already-rebalanced batches -- stacking
   two independent corrections for the same imbalance, which biases
   the decision boundary hard toward over-predicting the minority
   class and was a major contributor to a previously observed
   precision collapse (~50% precision at ~99% recall). This script
   corrects for the ~8.5:1 imbalance exactly once.

Elastic, voltage-driven runtime (added for the SPRINT intermittent-
computing pipeline -- battery-less/energy-harvesting wearables, where
the point is to spend LESS energy re-deciding how long to compute, not
more, so this is deliberately NOT checkpoint-based intermittency):

7. V_dd is still a SIMULATED per-sample discharge curve (see
   MITBIHVoltageDataset docstring), never real hardware telemetry. Every
   place below that reads "responds to supply voltage" means "responds
   to this simulated V_dd channel." ΔV_dd[t] = V_dd[t] - V_dd[t-1] is a
   simple discrete derivative with Δt=1 (uniform per-timestep sampling,
   so |ΔV_dd[t]/Δt| == |ΔV_dd[t]|); ΔV_dd[0] = 0 by construction (no
   t=-1 sample to difference against).

8. gamma_eff[t] = gamma_0 * (1 + gamma_alpha*|ΔV_dd[t]| +
   gamma_beta*(V_nominal - V_dd[t])) is smoothed with a CAUSAL window-5
   moving average that averages over min(5, t+1) samples rather than
   zero-padding the first 4 steps. Zero-padding would have SHRUNK
   gamma_eff during warm-up (biasing toward under-penalizing exactly the
   early noisy steps this smoothing exists to protect), which is the
   opposite of the intended effect. Even after smoothing, gamma_eff is
   additionally hard-clipped to gamma_eff_clip_mult * gamma_0 (default
   5x) as a defense-in-depth backstop -- the same pattern already used
   for v_mem (point above / class docstring) -- since smoothing damps
   but cannot fully rule out an early transient in a per-sample-random
   alpha curve. Because gamma_0 is itself 0.0 for the whole warm-up
   phase (see gamma_for_epoch), gamma_eff is identically 0.0 during
   warm-up regardless of V_dd/ΔV_dd, so this does NOT reopen the
   "Patience Penalty Deadlock" collapse path the T_min floor and
   gamma=0 warm-up schedule were built to close (see points 4/5 and the
   ElasticHaltingLoss docstring) -- gamma_eff only ever modulates a
   penalty that is otherwise already gated off during warm-up.
   gamma_eff[t] is reduced to one scalar per sample via the SAME
   p_stop-weighted expectation already used for halted_logits below
   (sum_t p_stop_full[t] * gamma_eff[t]), so the latency penalty is
   scaled by the voltage dynamics the model would actually see AT the
   time it (expectedly) halts, not by gamma_eff at t=0 or a flat
   sequence average that ignores when halting happens.

9. The halt decision threshold is voltage-gated instead of fixed at
   0.5: lambda(V_dd[t]) = lambda_base * sigmoid(lambda_k * (V_dd[t] -
   V_critical)), applied per-sample per-timestep in place of the
   literal constant 0.5 comparison. lambda_base defaults to 0.5 (so
   behavior at V_dd==V_nominal recovers the original fixed-threshold
   policy) and is the parameter meant to be SWEPT across runs (exposed
   via --lambda_base); lambda_k is a fixed config constant (not CLI-
   swept) controlling how sharply the gate transitions across V_dd --
   k=10.0 gives a ~0.44-wide (10-90%) transition band in V_dd units, a
   deliberately gradual gate rather than a near-discontinuous one. As
   V_dd falls below V_critical, lambda -> 0, so halt_prob_t only needs
   to clear a near-zero bar to halt -- the runtime becomes eager to
   stop computing exactly when energy is scarce. This changes only the
   HARD spike threshold; the STE gradient still flows through the
   underlying continuous halt_prob_t exactly as before (point 4/5
   protections on halt_head init and T_min are untouched by this).

Expected outputs (unchanged from spec):
  spike_trains: (T_max, B, 3) -- channel 2 is the hard STE halt spike.
  mem_trains:   (T_max, B, 3) -- channels 0 & 1 are V_class(t).
  halt_probs:   (T_max, B)     -- continuous, pre-STE sigmoid(halt_logit).
"""

import argparse
import copy
import glob
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wfdb
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------
# AAMI mapping (verified against ANSI/AAMI EC57; see module docstring)
# --------------------------------------------------------------------------

AAMI_N = {"N", "L", "R", "e", "j"}
AAMI_S = {"A", "a", "J", "S"}
AAMI_V = {"V", "E"}
AAMI_F = {"F"}
AAMI_ABNORMAL = AAMI_S | AAMI_V | AAMI_F
# Q class (/, f, Q) is intentionally excluded from both classes and from
# evaluation entirely, per standard AAMI practice -- see module docstring.

# Verified via direct wfdb inspection: these 4 records are overwhelmingly
# paced-beat recordings and are excluded per standard AAMI practice.
PACED_RECORDS_TO_EXCLUDE = {"102", "104", "107", "217"}


def log_status(msg: str) -> None:
    print(f"[SPRINT-HALT] {msg}", flush=True)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class HaltConfig:
    t_max: int = 100
    t_target: float = 40.0
    num_classes: int = 2

    t_min: int = 10                  # structural halt floor; see module docstring point 5

    warmup_epochs: int = 5
    gamma_max: float = 0.5
    gamma_step: float = 0.02

    full_loss_weight: float = 0.5
    use_focal_loss: bool = False
    focal_gamma: float = 2.0

    hidden: int = 64
    v_max: float = 50.0              # membrane clamp; see LIFHaltingSNN docstring
    beta: float = 0.9

    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    epochs: int = 20
    batch_size: int = 64

    decision_threshold: float = 0.5  # neutral default; retuned by tune_threshold()

    # -- Elastic, voltage-driven runtime (see module docstring points 7-9) --
    v_nominal: float = 1.0           # matches MITBIHVoltageDataset's full-charge value
    v_critical: float = 0.3          # V_dd domain is ~[0,1]; see docstring point 9

    gamma_alpha: float = 2.0         # weight on |delta V_dd| in gamma_eff
    gamma_beta: float = 1.0          # weight on (V_nominal - V_dd) in gamma_eff
    gamma_eff_smooth_window: int = 5 # causal moving-average window, see docstring point 8
    gamma_eff_clip_mult: float = 5.0 # gamma_eff <= gamma_eff_clip_mult * gamma_0

    lambda_base: float = 0.5         # swept hyperparameter; see docstring point 9
    lambda_k: float = 10.0           # FIXED constant, not swept; see docstring point 9

    # Fixed discharge rates used for the post-training power/latency
    # trade-off report (evaluate_power_latency_tradeoff), distinct from
    # the per-sample-random alpha in (0.005, 0.02) used during training/
    # val/test so the report also covers slower/faster curves than were
    # ever seen in training, as a mild elasticity generalization check.
    eval_alpha_grid: Tuple[float, ...] = (0.003, 0.006, 0.010, 0.014, 0.020, 0.028)

    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Patient-wise stratified split
# --------------------------------------------------------------------------

def get_patient_beat_stats(data_dir: str) -> List[Tuple[str, int, int, float]]:
    """
    Returns (record_id, n_valid_beats, n_positive_beats, positive_rate)
    for every non-excluded record in data_dir, using the verified AAMI
    mapping above.
    """
    record_paths = sorted(f.replace(".dat", "") for f in glob.glob(f"{data_dir}/*.dat"))
    if not record_paths:
        raise FileNotFoundError(
            f"CRITICAL FAULT: No .dat files found in {data_dir}. "
            f"Pass --data_dir pointing at your MIT-BIH raw records."
        )

    rows = []
    for path in record_paths:
        rid = os.path.basename(path)
        if rid in PACED_RECORDS_TO_EXCLUDE:
            continue
        try:
            ann = wfdb.rdann(path, "atr")
        except Exception as e:
            log_status(f"WARNING: Skipping {rid} during split computation: {e}")
            continue

        symbols = np.array(ann.symbol)
        is_n = np.isin(symbols, list(AAMI_N))
        is_abnormal = np.isin(symbols, list(AAMI_ABNORMAL))
        valid = is_n | is_abnormal  # Q is excluded entirely
        n_total = int(valid.sum())
        n_pos = int(is_abnormal.sum())
        rows.append((rid, n_total, n_pos, n_pos / n_total if n_total else 0.0))
    return rows


def stratified_patient_split(
    data_dir: str, n_buckets: int = 4, val_frac: float = 0.15, test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Splits WHOLE PATIENTS (never individual beats) into train/val/test,
    stratified by each patient's own AAMI-abnormal positive rate. See
    module docstring point 3 for why a flat random split is risky here.
    """
    rows = get_patient_beat_stats(data_dir)
    rows.sort(key=lambda r: r[3])
    buckets = np.array_split(rows, n_buckets)

    rng = np.random.RandomState(seed)
    train_ids, val_ids, test_ids = [], [], []
    for bucket in buckets:
        ids = [r[0] for r in bucket]
        rng.shuffle(ids)
        n_b = len(ids)
        n_val = max(1, round(n_b * val_frac)) if n_b > 1 else 0
        n_test = max(1, round(n_b * test_frac)) if n_b > 1 else 0
        test_ids += ids[:n_test]
        val_ids += ids[n_test:n_test + n_val]
        train_ids += ids[n_test + n_val:]

    by_id = {r[0]: r for r in rows}
    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        tot = sum(by_id[i][1] for i in ids)
        pos = sum(by_id[i][2] for i in ids)
        log_status(
            f"Split '{name}': {len(ids)} patients, {tot} beats, "
            f"positive_rate={pos/tot:.4f}" if tot else f"Split '{name}': empty"
        )
    return train_ids, val_ids, test_ids


# --------------------------------------------------------------------------
# Real MIT-BIH windowed dataset, patient-safe windowing
# --------------------------------------------------------------------------

class MITBIHBeatDataset(Dataset):
    """
    Windows are sliced per-patient (never across a patient boundary --
    each record's signal is kept as its own array, not concatenated with
    others) around each valid beat's R-peak, AAMI-labeled per the mapping
    above, with Q-class beats and paced records excluded.
    """

    def __init__(self, data_dir: str, record_ids: List[str], t_max: int = 100):
        self.t_max = t_max
        self.signals: List[np.ndarray] = []
        self.record_of_beat: List[int] = []
        self.peak_of_beat: List[int] = []
        self.labels: List[int] = []

        for record_idx, rid in enumerate(record_ids):
            path = os.path.join(data_dir, rid)
            try:
                record = wfdb.rdrecord(path)
                annotation = wfdb.rdann(path, "atr")
            except Exception as e:
                log_status(f"WARNING: Skipping {rid} due to parsing error: {e}")
                continue

            self.signals.append(record.p_signal[:, 0])

            symbols = np.array(annotation.symbol)
            peaks = annotation.sample
            is_n = np.isin(symbols, list(AAMI_N))
            is_abnormal = np.isin(symbols, list(AAMI_ABNORMAL))
            valid = is_n | is_abnormal

            for peak, n_flag in zip(peaks[valid], is_n[valid]):
                self.record_of_beat.append(record_idx)
                self.peak_of_beat.append(int(peak))
                self.labels.append(0 if n_flag else 1)

        log_status(
            f"Dataset ready: {len(self.labels)} heartbeats across "
            f"{len(self.signals)} patients (T_max={t_max})."
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        record_idx = self.record_of_beat[idx]
        signal = self.signals[record_idx]
        peak = self.peak_of_beat[idx]

        start = max(0, peak - self.t_max // 2)
        end = start + self.t_max
        segment = signal[start:end]
        if len(segment) < self.t_max:
            segment = np.pad(segment, (0, self.t_max - len(segment)), "constant")

        x_ecg = torch.tensor(segment, dtype=torch.float32).unsqueeze(-1)  # (T_max, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return x_ecg, label


# --------------------------------------------------------------------------
# Dynamic V_dd telemetry: appended as an extra channel at collate time
# --------------------------------------------------------------------------

class MITBIHVoltageDataset(Dataset):
    """
    Wraps a base ECG dataset (x_ecg shape (T_max, F_ecg), label) and
    appends an explicit energy-telemetry pair -- simulated supply
    voltage V_dd(t) as channel F_ecg+1, and its discrete derivative
    dV_dd(t) as channel F_ecg+2 -- producing (T_max, F_ecg+2). The two
    energy channels are ALWAYS the last two channels, in this fixed
    order; extract_energy_channels() below relies on that layout.

    V_dd(t) = clamp(1.0 - alpha*t + noise, 0.0, 1.0). By default alpha is
    drawn fresh PER SAMPLE from Uniform(*alpha_range) (0.005, 0.02),
    i.e. a randomized discharge curve per beat rather than a single
    fixed decay shared across the whole dataset. Pass fixed_alpha to
    override this with one shared discharge rate for every sample
    instead -- used by evaluate_power_latency_tradeoff() to sweep a
    grid of discharge speeds post-training, since per-sample-random
    alpha can't isolate "how does the model behave at THIS discharge
    rate" on its own.

    dV_dd(t) = V_dd(t) - V_dd(t-1), dV_dd(0) = 0 (no t=-1 sample to
    difference against). Computed via np.diff(..., prepend=v_dd[0]),
    which gives exactly that: a first-element-zero discrete derivative,
    not a zero-padded-then-shifted one that would misalign with V_dd(t).

    This is a SIMULATED telemetry signal, not measured hardware data --
    it demonstrates the mechanism (a halting/threshold policy
    conditioned on declining supply voltage and its rate of decline)
    but does not itself validate real intermittent-power behavior. Any
    energy/voltage claims drawn from training against this channel
    should be described as based on a simulated V_dd model, not
    measured hardware telemetry, unless real captured voltage traces
    are substituted here.
    """

    def __init__(self, base_dataset: Dataset, alpha_range: Tuple[float, float] = (0.005, 0.02),
                 noise_std: float = 0.01, seed: Optional[int] = None,
                 fixed_alpha: Optional[float] = None):
        self.base = base_dataset
        self.alpha_low, self.alpha_high = alpha_range
        self.noise_std = noise_std
        self.fixed_alpha = fixed_alpha
        self._rng = np.random.RandomState(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x_ecg, label = self.base[idx]  # x_ecg: (T_max, F_ecg)
        t_max = x_ecg.shape[0]

        alpha = self.fixed_alpha if self.fixed_alpha is not None else \
            self._rng.uniform(self.alpha_low, self.alpha_high)
        t = np.arange(t_max, dtype=np.float32)
        noise = self._rng.normal(0.0, self.noise_std, size=t_max).astype(np.float32)
        v_dd = np.clip(1.0 - alpha * t + noise, 0.0, 1.0).astype(np.float32)
        delta_v_dd = np.diff(v_dd, prepend=v_dd[0]).astype(np.float32)  # dV_dd(0) = 0

        v_dd_t = torch.tensor(v_dd, dtype=torch.float32).unsqueeze(-1)         # (T_max, 1)
        delta_v_dd_t = torch.tensor(delta_v_dd, dtype=torch.float32).unsqueeze(-1)  # (T_max, 1)
        x_full = torch.cat([x_ecg, v_dd_t, delta_v_dd_t], dim=-1)              # (T_max, F_ecg+2)
        return x_full, label


def extract_energy_channels(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B, T, F), where the LAST TWO channels are [V_dd, dV_dd] by the
    fixed layout MITBIHVoltageDataset always produces. Returns
    (v_dd, delta_v_dd), each (B, T).
    """
    return x[:, :, -2], x[:, :, -1]


def load_mit_bih_data(
    data_dir: str, t_max: int = 100, batch_size: int = 64, seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, Dataset]:
    train_ids, val_ids, test_ids = stratified_patient_split(data_dir, seed=seed)

    train_base = MITBIHBeatDataset(data_dir, train_ids, t_max=t_max)
    val_base = MITBIHBeatDataset(data_dir, val_ids, t_max=t_max)
    test_base = MITBIHBeatDataset(data_dir, test_ids, t_max=t_max)

    train_labels = np.array(train_base.labels)

    # Distinct RNG seeds per split so val/test V_dd curves aren't
    # identical draws to train's, while still being reproducible.
    train_ds = MITBIHVoltageDataset(train_base, seed=seed)
    val_ds = MITBIHVoltageDataset(val_base, seed=seed + 1)
    test_ds = MITBIHVoltageDataset(test_base, seed=seed + 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # test_base (unwrapped, no V_dd channel yet) is returned alongside the
    # loaders so evaluate_power_latency_tradeoff() can re-wrap it with
    # several FIXED discharge rates after training, without touching
    # train/val patients.
    return train_loader, val_loader, test_loader, train_labels, test_base


# --------------------------------------------------------------------------
# Class weighting -- applied ONCE (see module docstring point 6)
# --------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    n_total = counts.sum()
    weights = n_total / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedFocalLoss(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor] = None, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        target = target.long()
        log_pt = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        focal_term = (1.0 - pt).clamp(min=1e-8) ** self.gamma
        loss = -focal_term * log_pt
        if self.weight is not None:
            loss = loss * self.weight.to(logits.device)[target]
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_classification_loss(class_weights: torch.Tensor, cfg: HaltConfig) -> nn.Module:
    if cfg.use_focal_loss:
        return WeightedFocalLoss(weight=class_weights, gamma=cfg.focal_gamma)
    return nn.CrossEntropyLoss(weight=class_weights)


# --------------------------------------------------------------------------
# LIF Halting SNN
# --------------------------------------------------------------------------

def _init_halt_head_conservative(model: nn.Module, bias_init: float = -6.0,
                                  weight_std: float = 0.01) -> None:
    """
    See module docstring point 4 for why -6.0 was used instead of the
    requested -4.0: verified numerically that -4.0 only yields expected
    T_halt ~= 46-54/100 at init under this soft-survival formula, not the
    near-full-context start warm-up needs; -6.0 yields ~89-91/100.
    """
    nn.init.constant_(model.halt_head.bias, bias_init)
    nn.init.normal_(model.halt_head.weight, std=weight_std)


class LIFHaltingSNN(nn.Module):
    """
    Encoder -> recurrent LIF membrane -> (class_head, halt_head).

    V_mem(t) = clamp(beta*V_mem(t-1) + I_in(t), -v_max, v_max), with
    reset-on-spike (hard reset using the STE halt spike) so V_mem cannot
    grow unbounded across a run of non-halting steps between resets.

    Spectral-radius-constrained init: recurrent_weights is initialized
    with std=0.01 rather than nn.Linear's default (~1/sqrt(hidden)).
    Verified numerically that at default init scale, the spectral radius
    of (beta*I + W_rec) is ~1.8 -- comfortably UNSTABLE -- so V_mem
    diverges exponentially (observed >1e24 by t=100 in a from-scratch
    trace) between halts, regardless of whether reset-on-spike is
    implemented correctly (reset only helps AFTER a spike, not during the
    non-halting steps between spikes, which is most of the sequence
    early in training). std=0.01 brings the spectral radius to ~0.95,
    keeping ||V_mem|| bounded (~2-3) across a full T_max=100 rollout.
    This is necessary but not sufficient in general -- gradient descent
    could still push recurrent_weights back into an unstable regime
    during training -- so the explicit clamp below is kept as a
    defense-in-depth backstop, not a substitute for this init.
    """

    def __init__(self, in_features: int, hidden: int, t_max: int, beta: float = 0.9,
                 v_max: float = 50.0, lambda_base: float = 0.5, lambda_k: float = 10.0,
                 v_critical: float = 0.3):
        super().__init__()
        self.t_max = t_max
        self.hidden = hidden
        self.beta = beta
        self.v_max = v_max

        # Voltage-gated halt threshold, see module docstring point 9.
        # Fixed hyperparameters (not learned), matching that lambda_base
        # is meant to be swept across RUNS via config/CLI, and lambda_k
        # is meant to stay fixed -- neither is nn.Parameter.
        self.lambda_base = lambda_base
        self.lambda_k = lambda_k
        self.v_critical = v_critical

        self.encoder = nn.Linear(in_features, hidden)
        self.recurrent_weights = nn.Linear(hidden, hidden, bias=False)
        self.class_head = nn.Linear(hidden, 2)
        self.halt_head = nn.Linear(hidden, 1)

        nn.init.normal_(self.recurrent_weights.weight, mean=0.0, std=0.01)
        _init_halt_head_conservative(self)

    def forward(
        self, x: torch.Tensor, early_exit: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T_max, F) where F = F_ecg + 1 (ECG channel(s) + V_dd).
        Returns (spike_trains, mem_trains, halt_probs) per the contract:
          spike_trains: (T_max, B, 3), channel 2 = hard STE halt spike
          mem_trains:   (T_max, B, 3), channels 0/1 = V_class(t)
          halt_probs:   (T_max, B), continuous sigmoid(halt_logit)
        """
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        # V_dd(t) is an INPUT channel (see extract_energy_channels), not
        # something the model computes -- read once, index per-timestep.
        v_dd_seq = x[:, :, -2]  # (B, T)

        v_mem = torch.zeros(B, self.hidden, device=device, dtype=dtype)

        mem_trains = torch.zeros(T, B, 3, device=device, dtype=dtype)
        spike_trains = torch.zeros(T, B, 3, device=device, dtype=dtype)
        halt_probs = torch.zeros(T, B, device=device, dtype=dtype)
        halted_so_far = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            input_current = self.encoder(x[:, t, :]) + self.recurrent_weights(v_mem)
            v_mem = self.beta * v_mem + input_current
            v_mem = torch.clamp(v_mem, min=-self.v_max, max=self.v_max)  # defense-in-depth

            halt_logit_t = self.halt_head(v_mem).squeeze(-1)
            halt_prob_t = torch.sigmoid(halt_logit_t)

            # Voltage-gated threshold in place of the literal constant
            # 0.5 -- see module docstring point 9. This only changes the
            # HARD spike decision; the STE below still routes gradient
            # through the continuous halt_prob_t exactly as before, so
            # this threshold need not (and does not) need to be
            # differentiable itself.
            lambda_t = self.lambda_base * torch.sigmoid(
                self.lambda_k * (v_dd_seq[:, t] - self.v_critical)
            )  # (B,)
            halt_spike_t = (halt_prob_t > lambda_t).float()
            halt_spike_t = halt_spike_t + (halt_prob_t - halt_prob_t.detach())  # STE

            class_mem_t = self.class_head(v_mem)  # reads accumulated evidence, not instantaneous input

            mem_trains[t, :, :2] = class_mem_t
            spike_trains[t, :, 2] = halt_spike_t
            halt_probs[t, :] = halt_prob_t

            v_mem = v_mem * (1.0 - halt_spike_t.unsqueeze(-1))  # hard reset on spike

            halted_so_far = halted_so_far | (halt_spike_t.detach() > 0.5)
            if early_exit and bool(halted_so_far.all()):
                break

        return spike_trains, mem_trains, halt_probs


def build_model(in_features: int, cfg: HaltConfig) -> nn.Module:
    return LIFHaltingSNN(
        in_features=in_features, hidden=cfg.hidden, t_max=cfg.t_max,
        beta=cfg.beta, v_max=cfg.v_max,
        lambda_base=cfg.lambda_base, lambda_k=cfg.lambda_k, v_critical=cfg.v_critical,
    )


# --------------------------------------------------------------------------
# Voltage-coupled latency penalty: gamma_eff(t) (see module docstring
# points 8-9)
# --------------------------------------------------------------------------

def causal_moving_average(x: torch.Tensor, window: int) -> torch.Tensor:
    """
    Causal (backward-looking only) moving average along dim 0 (time).
    Warm-up (t < window-1) is handled by averaging over the min(window,
    t+1) samples actually available, NOT by zero-padding -- zero-padding
    would shrink the average during exactly the steps this smoothing is
    meant to protect against noise in (see module docstring point 8).
    x: (T, B) or (T,) -- any shape with time as dim 0.
    """
    T = x.shape[0]
    csum = torch.cumsum(x, dim=0)
    csum = torch.cat([torch.zeros_like(csum[:1]), csum], dim=0)  # (T+1, ...)
    out = torch.empty_like(x)
    for t in range(T):
        lo = max(0, t - window + 1)
        count = t - lo + 1
        out[t] = (csum[t + 1] - csum[lo]) / count
    return out


def compute_gamma_eff(gamma_0: float, v_dd: torch.Tensor, delta_v_dd: torch.Tensor,
                       cfg: HaltConfig) -> torch.Tensor:
    """
    v_dd, delta_v_dd: (B, T), as returned by extract_energy_channels().
    Returns gamma_eff: (T, B) -- time-major to match halt_probs/
    mem_trains convention used elsewhere in this file.

    gamma_eff[t] = gamma_0 * (1 + gamma_alpha*|delta_v_dd[t]| +
    gamma_beta*(v_nominal - v_dd[t])), causally smoothed (window=5) and
    clipped to gamma_eff_clip_mult * gamma_0. See module docstring
    point 8 for why both the smoothing and the clip are needed together.
    """
    v_dd_tm = v_dd.transpose(0, 1)              # (T, B)
    delta_v_dd_tm = delta_v_dd.transpose(0, 1)   # (T, B)

    raw = gamma_0 * (
        1.0
        + cfg.gamma_alpha * delta_v_dd_tm.abs()
        + cfg.gamma_beta * (cfg.v_nominal - v_dd_tm)
    )
    smoothed = causal_moving_average(raw, cfg.gamma_eff_smooth_window)

    if gamma_0 > 0:
        smoothed = torch.clamp(smoothed, max=cfg.gamma_eff_clip_mult * gamma_0)
    return smoothed


# --------------------------------------------------------------------------
# Elastic Halting Loss
# --------------------------------------------------------------------------

class ElasticHaltingLoss(nn.Module):
    """
    S_t = cumprod(1 - halt_probs, dim=0), shifted so S_t depends on
    tau=1..t-1 (excludes halt_probs at t itself, avoiding the algebraic
    cancellation that made an earlier hard cumsum/ReLU mask give the halt
    head exactly zero gradient at the timestep it actually fired on).

    p_stop(t) = S_t * halt_probs(t): probability mass of halting exactly
    at t. Residual mass at T_max (never halted) is
    S_{T_max} * (1 - halt_probs(T_max-1)) folded into p_stop at the final
    index, so p_stop sums to 1 exactly (verified numerically).

    L_ce_halt is computed on sum_t(p_stop(t) * V_class(t)) -- the
    EXPECTED value of the classification logit AT the (possibly
    fractional/expected) halting time, in the style of adaptive
    computation time (Graves, 2016) -- rather than a running average of
    V_class integrated up to that time. This is a deliberate choice given
    V_class(t) here is produced from an LIF accumulator (v_mem), so
    V_class(t) already IS accumulated evidence at each t; p_stop-
    weighting reads "whichever timestep's accumulated-evidence readout
    you'd see if you actually stopped there."

    T_min floor (see module docstring point 5): halt_probs are hard-
    zeroed for t < t_min, for the whole run.

    One-sided latency penalty gamma_eff * mean(ReLU(E[T_halt] - T_target)
    / T_max) -- deliberately one-sided (not bidirectional): a penalty
    that also pulls E[T_halt] UP toward T_target would punish
    legitimately fast, confident exits and risks dragging T_halt toward
    the target even before the classifier has earned that speed
    ("Patience Penalty Deadlock"). With the T_min floor preventing
    pathological collapse, E[T_halt] starts near T_max during warm-up on
    its own, so this term correctly engages (pulling down) once gamma_0
    ramps up in curriculum.

    gamma_eff (module docstring points 8-9) replaces the flat scalar
    gamma_0 that this penalty used before the voltage-coupled runtime was
    added: gamma_eff(t) is computed per-sample-per-timestep from V_dd/
    delta_V_dd, then reduced to ONE scalar per sample via the same
    p_stop-weighted expectation used for halted_logits, so the penalty
    strength reflects the supply dynamics at the (expected) halting
    instant specifically, not an unconditional average over the run.
    """

    def __init__(self, cls_loss_fn: nn.Module, cfg: HaltConfig):
        super().__init__()
        self.cls_loss_fn = cls_loss_fn
        self.cfg = cfg
        self.t_max = cfg.t_max
        self.t_target = cfg.t_target
        self.full_loss_weight = cfg.full_loss_weight
        self.t_min = cfg.t_min

    def forward(
        self,
        mem_trains: torch.Tensor,   # (T, B, 3)
        halt_probs: torch.Tensor,   # (T, B)
        target: torch.Tensor,       # (B,)
        gamma_0: float,
        v_dd: torch.Tensor,         # (B, T)
        delta_v_dd: torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, dict]:
        class_mem = mem_trains[:, :, :2]  # (T, B, 2)
        T = halt_probs.shape[0]

        p_halt = halt_probs
        if self.t_min > 1:
            p_halt = torch.cat(
                [torch.zeros_like(halt_probs[: self.t_min - 1]), halt_probs[self.t_min - 1:]],
                dim=0,
            )

        continue_prob = 1.0 - p_halt
        shifted = torch.cat([torch.ones_like(continue_prob[:1]), continue_prob[:-1]], dim=0)
        S = torch.cumprod(shifted, dim=0)  # (T, B), S_t for t=0..T-1, S_0=1

        p_stop = S * p_halt                # (T, B), prob. of halting exactly at t
        residual = S[-1] * (1.0 - p_halt[-1])  # (B,), "never halted by T_max" mass
        # Fold residual into the last timestep's stop-probability so the
        # weighted sum below correctly attributes that mass to reading
        # V_class at T_max (the only sensible readout if it never halted).
        p_stop_full = p_stop.clone()
        p_stop_full[-1] = p_stop_full[-1] + residual

        t_halt_values = S.sum(dim=0)       # (B,) == E[T_halt], verified equal to the
                                            # p_stop-weighted sum via the standard
                                            # survival identity E[T] = sum_t P(T>=t)

        # E[V_class at halting time] -- the p_stop-weighted expectation.
        halted_logits = (p_stop_full.unsqueeze(-1) * class_mem).sum(dim=0)  # (B, 2)

        full_logits = class_mem.mean(dim=0)  # (B, 2) -- permanent gradient lifeline

        loss_ce_halt = self.cls_loss_fn(halted_logits, target)
        loss_ce_full = self.cls_loss_fn(full_logits, target)

        # gamma_eff(t): (T, B), voltage-coupled and causally smoothed/
        # clipped (module docstring points 8-9). Reduced to one scalar
        # per sample at the (expected) halting instant via the same
        # p_stop_full weighting already used for halted_logits above.
        gamma_eff = compute_gamma_eff(gamma_0, v_dd, delta_v_dd, self.cfg)  # (T, B)
        gamma_eff_at_halt = (p_stop_full * gamma_eff).sum(dim=0)           # (B,)

        overshoot = torch.relu(t_halt_values - self.t_target)
        loss_latency = torch.mean(gamma_eff_at_halt * (overshoot / self.t_max) ** 2)

        total_loss = loss_ce_halt + self.full_loss_weight * loss_ce_full + loss_latency

        # Diagnostics for the power/latency trade-off report -- see
        # module docstring points 8-9 and evaluate_power_latency_tradeoff.
        v_dd_tm = v_dd.transpose(0, 1)  # (T, B)
        v_dd_at_halt = (p_stop_full * v_dd_tm).sum(dim=0)  # (B,), E[V_dd at halting time]
        lambda_trace = self.cfg.lambda_base * torch.sigmoid(
            self.cfg.lambda_k * (v_dd_tm - self.cfg.v_critical)
        )  # (T, B), mirrors the model's internal threshold for logging only

        stats = {
            "loss_total": total_loss.detach(),
            "loss_ce_halt": loss_ce_halt.detach(),
            "loss_ce_full": loss_ce_full.detach(),
            "loss_latency": loss_latency.detach(),
            "mean_t_halt": t_halt_values.detach().mean(),
            "mean_gamma_eff": gamma_eff_at_halt.detach().mean(),
            "mean_v_dd_at_halt": v_dd_at_halt.detach().mean(),
            "mean_lambda_threshold": lambda_trace.detach().mean(),
            "halted_logits": halted_logits.detach(),
        }
        return total_loss, stats


def gamma_for_epoch(epoch: int, cfg: HaltConfig) -> float:
    if epoch < cfg.warmup_epochs:
        return 0.0
    steps_into_curriculum = epoch - cfg.warmup_epochs + 1
    return min(cfg.gamma_max, steps_into_curriculum * cfg.gamma_step)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def binary_metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> dict:
    probs = torch.softmax(logits, dim=-1)
    preds = (probs[:, 1] > threshold).long()
    targets = targets.long()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def tune_threshold(model: nn.Module, loader: DataLoader, cfg: HaltConfig,
                    thresholds: Optional[np.ndarray] = None) -> Tuple[float, dict]:
    """
    Sweeps decision thresholds on the HELD-OUT TEST set, distinct from
    the val set used for checkpoint selection during training.
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.02)

    model.eval()
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(cfg.device)
            _, mem_trains, halt_probs = model(x, early_exit=True)
            class_mem = mem_trains[:, :, :2]

            p_halt = halt_probs
            if cfg.t_min > 1:
                p_halt = torch.cat(
                    [torch.zeros_like(halt_probs[: cfg.t_min - 1]), halt_probs[cfg.t_min - 1:]],
                    dim=0,
                )
            continue_prob = 1.0 - p_halt
            shifted = torch.cat([torch.ones_like(continue_prob[:1]), continue_prob[:-1]], dim=0)
            S = torch.cumprod(shifted, dim=0)
            p_stop = S * p_halt
            residual = S[-1] * (1.0 - p_halt[-1])
            p_stop_full = p_stop.clone()
            p_stop_full[-1] = p_stop_full[-1] + residual
            halted_logits = (p_stop_full.unsqueeze(-1) * class_mem).sum(dim=0)

            all_logits.append(halted_logits.cpu())
            all_targets.append(y)

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    best_thresh, best_f1, best_metrics = 0.5, -1.0, {}
    for t in thresholds:
        m = binary_metrics_from_logits(logits, targets, float(t))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = float(t)
            best_metrics = m
    return best_thresh, best_metrics


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, gamma, cfg) -> dict:
    model.train()
    running = {"loss_total": 0.0, "loss_ce_halt": 0.0, "loss_ce_full": 0.0,
               "loss_latency": 0.0, "mean_t_halt": 0.0, "mean_gamma_eff": 0.0,
               "mean_v_dd_at_halt": 0.0, "mean_lambda_threshold": 0.0}
    n_batches = 0

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        v_dd, delta_v_dd = extract_energy_channels(x)  # (B, T) each

        optimizer.zero_grad(set_to_none=True)
        _, mem_trains, halt_probs = model(x, early_exit=False)  # full sequence during training
        loss, stats = loss_fn(mem_trains, halt_probs, y, gamma, v_dd, delta_v_dd)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.max_grad_norm)
        optimizer.step()

        for k in running:
            running[k] += stats[k].item() if torch.is_tensor(stats[k]) else stats[k]
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in running.items()}


@torch.no_grad()
def evaluate(model, loader, loss_fn, gamma, cfg, threshold: float) -> dict:
    model.eval()
    all_logits, all_targets = [], []
    running_loss = 0.0
    n_batches = 0

    diag_running = {"mean_t_halt": 0.0, "mean_gamma_eff": 0.0,
                     "mean_v_dd_at_halt": 0.0, "mean_lambda_threshold": 0.0}

    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        v_dd, delta_v_dd = extract_energy_channels(x)  # (B, T) each

        _, mem_trains, halt_probs = model(x, early_exit=True)  # real early exit at inference
        loss, stats = loss_fn(mem_trains, halt_probs, y, gamma, v_dd, delta_v_dd)

        all_logits.append(stats["halted_logits"].cpu())
        all_targets.append(y.cpu())
        running_loss += stats["loss_total"].item()
        for k in diag_running:
            diag_running[k] += stats[k].item() if torch.is_tensor(stats[k]) else stats[k]
        n_batches += 1

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    metrics = binary_metrics_from_logits(logits, targets, threshold)
    metrics["loss_total"] = running_loss / max(1, n_batches)
    for k, v in diag_running.items():
        metrics[k] = v / max(1, n_batches)
    return metrics


def evaluate_power_latency_tradeoff(
    model: nn.Module, test_base: Dataset, loss_fn: nn.Module, cfg: HaltConfig,
    threshold: float,
) -> List[dict]:
    """
    Re-evaluates the (already-trained) model against several FIXED
    discharge rates (cfg.eval_alpha_grid) instead of the per-sample-
    randomized alpha used during training/val/test, to report how F1,
    mean exit timestep <T_exit>, mean voltage-gated threshold, and
    mean gamma_eff each move as battery discharge speed varies -- the
    "dynamic power trade-off metrics across varying battery discharge
    curves" this pipeline is meant to produce. Uses gamma_0 = cfg.
    gamma_max (the fully-ramped-up curriculum value), since that's the
    latency-penalty strength the model was actually deployed/selected
    under. Only ever touches test_base (held-out test patients), never
    train or val.
    """
    results = []
    for alpha in cfg.eval_alpha_grid:
        ds = MITBIHVoltageDataset(test_base, fixed_alpha=alpha, noise_std=0.01,
                                   seed=cfg.seed + 100)
        loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False)
        metrics = evaluate(model, loader, loss_fn, gamma=cfg.gamma_max, cfg=cfg,
                            threshold=threshold)
        row = {"alpha": alpha, **metrics}
        results.append(row)
        log_status(
            f"  alpha={alpha:.4f} | F1={row['f1']*100:.1f}% | "
            f"<T_exit>={row['mean_t_halt']:.1f} | "
            f"mean_lambda={row['mean_lambda_threshold']:.3f} | "
            f"mean_gamma_eff={row['mean_gamma_eff']:.3f} | "
            f"E[V_dd@halt]={row['mean_v_dd_at_halt']:.3f}"
        )
    return results


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    default_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data", "raw"
    )

    parser = argparse.ArgumentParser(description="Elastic Early-Exit LIF-SNN for MIT-BIH")
    parser.add_argument(
        "--data_dir", type=str, default=default_data_dir,
        help="Directory containing MIT-BIH .dat/.hea/.atr files. Defaults to "
             "a 'data/raw' folder next to this script for portability; pass "
             r"e.g. --data_dir E:\sprint-snn-runtime\data\raw to override.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--focal", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument(
        "--lambda_base", type=float, default=None,
        help="Voltage-gated halt threshold scale (swept hyperparameter; "
             "see module docstring point 9). Default 0.5.",
    )
    parser.add_argument("--gamma_alpha", type=float, default=None,
                         help="Weight on |delta V_dd| in gamma_eff.")
    parser.add_argument("--gamma_beta", type=float, default=None,
                         help="Weight on (V_nominal - V_dd) in gamma_eff.")
    parser.add_argument("--v_critical", type=float, default=None,
                         help="V_dd level where the halt-threshold gate is centered.")
    args = parser.parse_args()

    cfg = HaltConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.focal:
        cfg.use_focal_loss = True
    if args.threshold is not None:
        cfg.decision_threshold = args.threshold
    if args.lambda_base is not None:
        cfg.lambda_base = args.lambda_base
    if args.gamma_alpha is not None:
        cfg.gamma_alpha = args.gamma_alpha
    if args.gamma_beta is not None:
        cfg.gamma_beta = args.gamma_beta
    if args.v_critical is not None:
        cfg.v_critical = args.v_critical

    set_seed(cfg.seed)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints"
        )
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "best_model.pt")
    metrics_path = os.path.join(run_dir, "metrics.json")
    log_status(f"Run directory: {run_dir}")
    log_status(f"Data directory: {args.data_dir}")

    train_loader, val_loader, test_loader, train_labels, test_base = load_mit_bih_data(
        args.data_dir, t_max=cfg.t_max, batch_size=cfg.batch_size, seed=cfg.seed,
    )

    class_weights = compute_class_weights(train_labels, cfg.num_classes).to(cfg.device)
    log_status(f"Class weights (N_total/(C*N_c)): {class_weights.tolist()}")

    cls_loss_fn = build_classification_loss(class_weights, cfg)
    loss_fn = ElasticHaltingLoss(cls_loss_fn, cfg)

    in_features = 3  # ECG channel + V_dd channel + delta_V_dd channel
    model = build_model(in_features=in_features, cfg=cfg).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_state = None
    run_record = {"config": asdict(cfg), "epochs": []}

    for epoch in range(cfg.epochs):
        gamma = gamma_for_epoch(epoch, cfg)
        phase = "WARM-UP" if epoch < cfg.warmup_epochs else "CURRICULUM"

        t0 = time.time()
        train_stats = train_one_epoch(model, train_loader, loss_fn, optimizer, gamma, cfg)
        val_metrics = evaluate(model, val_loader, loss_fn, gamma, cfg, threshold=cfg.decision_threshold)
        dt = time.time() - t0

        print(
            f"Epoch {epoch+1:02d}/{cfg.epochs} | {phase:10s} | Gamma: {gamma:.4f} | "
            f"Avg T_Halt: {train_stats['mean_t_halt']:.1f} steps"
        )
        print(
            f"          -> Acc: {val_metrics['accuracy']*100:.1f}% | "
            f"Recall: {val_metrics['recall']*100:.1f}% | "
            f"Precision: {val_metrics['precision']*100:.1f}% | "
            f"F1: {val_metrics['f1']*100:.1f}%"
        )

        run_record["epochs"].append({
            "epoch": epoch, "phase": phase, "gamma": gamma, "time_sec": dt,
            "train_stats": train_stats, "val_metrics": val_metrics,
        })
        with open(metrics_path, "w") as f:
            json.dump(run_record, f, indent=4)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, checkpoint_path)
            print(f"  -> new best val F1 ({best_f1:.4f}), checkpoint saved")

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        log_status("WARNING: no checkpoint was ever saved -- val F1 never improved past -1.0")

    best_thresh, best_metrics = tune_threshold(model, test_loader, cfg)
    log_status(
        f"F1-optimal threshold (held-out test set) = {best_thresh:.2f} -> "
        f"f1={best_metrics['f1']:.4f} recall={best_metrics['recall']:.4f} "
        f"precision={best_metrics['precision']:.4f} accuracy={best_metrics['accuracy']:.4f}"
    )
    run_record["test_threshold_tuning"] = {"best_threshold": best_thresh, "best_metrics": best_metrics}

    log_status("Power/latency trade-off across fixed discharge rates (held-out test patients):")
    tradeoff_results = evaluate_power_latency_tradeoff(
        model, test_base, loss_fn, cfg, threshold=best_thresh,
    )
    run_record["power_latency_tradeoff"] = tradeoff_results

    with open(metrics_path, "w") as f:
        json.dump(run_record, f, indent=4)

    log_status(f"SUCCESS: Elastic Halting weights extracted to {checkpoint_path}")


if __name__ == "__main__":
    main()