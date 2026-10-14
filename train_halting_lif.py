"""
train_halting_lif.py -- spiking variant of the SPRINT halting model.

Reuses everything in train_halting_snn.py (data, split, loss, curriculum,
CLI flags, checkpointing) and swaps only the model. Run it exactly like the
original, e.g.:

  python train_halting_lif.py --data_dir data/raw --epochs 15 --lr 1e-3

Changes relative to LIFHaltingSNN:
  1. Real LIF hidden layer: Heaviside spike at threshold theta, fast-sigmoid
     surrogate gradient (same family as snnTorch's surrogate.fast_sigmoid),
     soft reset. Recurrence is driven by binary spikes, so on neuromorphic
     hardware the hidden and readout layers cost accumulates (SOPs), not MACs.
  2. V_dd and dV_dd feed ONLY the halt head. The class pathway sees ECG alone,
     so changing the discharge rate can move WHEN the model exits but cannot
     shift the class logits themselves.
  3. Causal baseline removal: ECG is referenced to the first sample of the
     window (no future samples used, so early exit stays honest).
  4. Readout from a leaky synaptic trace of the hidden spikes.

Per-timestep hidden spike counts are kept in model.last_spikes (T, B) after
each forward pass; eval_sprint.py uses them to report SOPs up to the exit.
"""

import torch
import torch.nn as nn

import train_halting_snn as ths


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return (x > 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return grad_out / (ctx.slope * x.abs() + 1.0) ** 2, None


class SpikingHaltingSNN(ths.LIFHaltingSNN):
    def __init__(self, hidden, t_max, beta=0.9, v_max=50.0, lambda_base=0.5,
                 lambda_k=10.0, v_critical=0.3, t_min=10, v_emergency=0.05,
                 theta=1.0, slope=25.0):
        super().__init__(in_features=1, hidden=hidden, t_max=t_max, beta=beta, v_max=v_max,
                         lambda_base=lambda_base, lambda_k=lambda_k, v_critical=v_critical,
                         t_min=t_min, v_emergency=v_emergency)
        self.theta = theta
        self.slope = slope
        # Halt head sees the spike trace plus the two energy channels.
        self.halt_head = nn.Linear(hidden + 2, 1)
        ths._init_halt_head_conservative(self)
        self.last_spikes = None

    def forward(self, x, early_exit=False, voltage_gate=1.0):
        B, T, _ = x.shape
        device, dtype = x.device, x.dtype

        ecg = x[:, :, :1] - x[:, :1, :1]   # causal baseline removal
        v_dd_seq = x[:, :, -2]
        dv_dd_seq = x[:, :, -1]

        v = torch.zeros(B, self.hidden, device=device, dtype=dtype)
        spk = torch.zeros(B, self.hidden, device=device, dtype=dtype)
        trace = torch.zeros(B, self.hidden, device=device, dtype=dtype)

        mem_trains = torch.zeros(T, B, 3, device=device, dtype=dtype)
        spike_trains = torch.zeros(T, B, 3, device=device, dtype=dtype)
        halt_probs = torch.zeros(T, B, device=device, dtype=dtype)
        spikes_per_step = torch.zeros(T, B, device=device, dtype=dtype)
        halted_so_far = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(T):
            v = self.beta * v + self.encoder(ecg[:, t]) + self.recurrent_weights(spk)
            spk = FastSigmoidSpike.apply(v - self.theta, self.slope)
            v = torch.clamp(v - spk * self.theta, -self.v_max, self.v_max)  # soft reset
            trace = self.beta * trace + spk
            spikes_per_step[t] = spk.detach().sum(-1)

            halt_in = torch.cat([trace, v_dd_seq[:, t:t + 1], dv_dd_seq[:, t:t + 1]], dim=-1)
            halt_logit_t = torch.clamp(self.halt_head(halt_in).squeeze(-1), -15.0, 15.0)
            halt_prob_t = torch.sigmoid(halt_logit_t)

            if t < self.t_min:
                halt_spike_t = torch.zeros(B, device=device, dtype=dtype)
            else:
                lambda_t = self.lambda_base * (
                    (1.0 - voltage_gate)
                    + voltage_gate * torch.sigmoid(self.lambda_k * (v_dd_seq[:, t] - self.v_critical))
                )
                halt_spike_t = (halt_prob_t > lambda_t).float()
                halt_spike_t = halt_spike_t + (halt_prob_t - halt_prob_t.detach())  # STE
                emergency = (v_dd_seq[:, t] < self.v_emergency).float().detach()
                halt_spike_t = torch.maximum(halt_spike_t, emergency)

            mem_trains[t, :, :2] = self.class_head(trace)
            spike_trains[t, :, 2] = halt_spike_t
            halt_probs[t, :] = halt_prob_t

            keep = (1.0 - halt_spike_t).unsqueeze(-1)  # hard reset on halt, as in the original
            v = v * keep
            trace = trace * keep

            halted_so_far = halted_so_far | (halt_spike_t.detach() > 0.5)
            if early_exit and bool(halted_so_far.all()):
                break

        self.last_spikes = spikes_per_step
        return spike_trains, mem_trains, halt_probs


def build_model(in_features, cfg):
    return SpikingHaltingSNN(
        hidden=cfg.hidden, t_max=cfg.t_max, beta=cfg.beta, v_max=cfg.v_max,
        lambda_base=cfg.lambda_base, lambda_k=cfg.lambda_k, v_critical=cfg.v_critical,
        t_min=cfg.t_min, v_emergency=cfg.v_emergency,
    )


_HaltConfigClass = ths.HaltConfig   # keep the real class; main() gets a factory below


def _config_from_json(path):
    """Start from the exact HaltConfig of an earlier run (its metrics.json),
    so the retrain differs only in the model. CLI flags still override it."""
    import dataclasses
    import json
    with open(path) as f:
        saved = json.load(f).get("config", {})
    cfg = _HaltConfigClass()
    names = {fld.name for fld in dataclasses.fields(_HaltConfigClass)}
    for k, v in saved.items():
        if k in names and k != "device":
            setattr(cfg, k, tuple(v) if isinstance(v, list) else v)
    return cfg


if __name__ == "__main__":
    import sys
    if "--config_from" in sys.argv:
        i = sys.argv.index("--config_from")
        path = sys.argv[i + 1]
        del sys.argv[i:i + 2]
        ths.HaltConfig = lambda: _config_from_json(path)   # main() calls HaltConfig() once
        print(f"[SPRINT-LIF] Base config loaded from {path}", flush=True)
    ths.build_model = build_model   # main() looks this name up at call time
    ths.main()
