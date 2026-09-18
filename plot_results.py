"""
generate_all_plots.py

Generates publication-ready IEEE benchmark figures from metrics.json and 
analyzes model weight distributions from a PyTorch checkpoint file.

Usage:
    python generate_all_plots.py --metrics_json metrics.json --weights_file model.pt --out_dir figures/
"""

import argparse
import glob
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# Optional PyTorch import for weights analysis
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Global IEEE Style Configuration
# --------------------------------------------------------------------------

_WINDOWS_FONT_DIRS = [
    r"C:\Windows\Fonts",
    "/mnt/c/Windows/Fonts",
    "/usr/share/fonts/truetype/msttcorefonts",
]


def _register_times_new_roman() -> None:
    candidates = ["times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf", "Times_New_Roman.ttf"]
    for font_dir in _WINDOWS_FONT_DIRS:
        if not os.path.isdir(font_dir):
            continue
        for fname in candidates:
            for path in set(glob.glob(os.path.join(font_dir, fname)) +
                             glob.glob(os.path.join(font_dir, fname.upper()))):
                fm.fontManager.addfont(path)


_register_times_new_roman()

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "font.weight": "bold",
    "axes.labelsize": 10.0,
    "axes.labelweight": "bold",
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.linewidth": 1.5,
    "axes.edgecolor": "black",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "grid.linestyle": ":",
    "grid.color": "#888888",
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handletextpad": 0.4,
    "lines.markersize": 4.5,
    "lines.markeredgewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGSIZE_SINGLE = (3.4, 2.2)

SERIES_STYLES = {
    "precision": dict(color="#008000", marker="o", ls="-"),
    "recall":    dict(color="#B22222", marker="s", ls="--"),
    "f1":        dict(color="#00008B", marker="^", ls="-."),
    "accuracy":  dict(color="#4B0082", marker="D", ls=":"),
}


def load_metrics(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color("black")
    ax.tick_params(colors="black", width=1.2, which="both", pad=2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def savefig(fig, out_dir: str, name: str) -> None:
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(out_dir, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{name}.png"), bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved {name}.pdf / {name}.png")


# --------------------------------------------------------------------------
# Figure 1: Training & Validation Metrics Over Epochs
# --------------------------------------------------------------------------

def plot_training_curves(d: dict, out_dir: str) -> None:
    epochs = d.get("epochs", [])
    if not epochs:
        print("[!] Warning: 'epochs' key missing in metrics.json. Skipping Fig 1.")
        return

    ep_idx = [e["epoch"] for e in epochs]
    prec = [e["val_metrics"]["precision"] * 100 for e in epochs]
    rec = [e["val_metrics"]["recall"] * 100 for e in epochs]
    f1 = [e["val_metrics"]["f1"] * 100 for e in epochs]
    warmup_epochs = sum(1 for e in epochs if e.get("phase") == "WARM-UP")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    if warmup_epochs > 0:
        ax.axvspan(min(ep_idx), warmup_epochs, color="0.92", zorder=0)
        ax.axvline(warmup_epochs, color="0.5", ls="--", lw=0.8, zorder=1)

    s = SERIES_STYLES["precision"]
    ax.plot(ep_idx, prec, color=s["color"], marker=s["marker"], ls=s["ls"],
            lw=1.3, mfc="none", mew=1.0, label="Precision")
    s = SERIES_STYLES["recall"]
    ax.plot(ep_idx, rec, color=s["color"], marker=s["marker"], ls=s["ls"],
            lw=1.3, mfc="none", mew=1.0, label="Recall")
    s = SERIES_STYLES["f1"]
    ax.plot(ep_idx, f1, color=s["color"], marker=s["marker"], ls=s["ls"],
            lw=1.3, mfc="none", mew=1.0, label="F1")

    ax.set_title("TRAINING METRICS", pad=4)
    ax.set_xlabel("Epoch", labelpad=2)
    ax.set_ylabel("Validation Metric (%)", labelpad=2)
    ax.set_xlim(min(ep_idx), max(ep_idx))
    ax.set_ylim(0, 102)

    leg = ax.legend(loc="lower right", handlelength=1.8)
    leg.get_frame().set_linewidth(1.0)
    style_axes(ax)
    savefig(fig, out_dir, "fig1_training_curves")


# --------------------------------------------------------------------------
# Figure 2: Recall at Precision Floor Constraint
# --------------------------------------------------------------------------

def plot_precision_floor(d: dict, out_dir: str) -> None:
    epochs = d.get("epochs", [])
    if not epochs:
        return

    precision_floor = d.get("config", {}).get("precision_floor", 0.90)
    ep_idx, pf_rec, cleared = [], [], []

    for e in epochs:
        pf = e.get("val_precision_floor_check", {})
        m = pf.get("metrics", {})
        ep_idx.append(e["epoch"])
        if pf.get("threshold") is not None:
            pf_rec.append(m.get("recall", 0.0) * 100)
            cleared.append(True)
        else:
            pf_rec.append(m.get("precision", 0.0) * 100)
            cleared.append(False)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bar_color = "#1B4F72"
    hatch_uncleared = "///"
    colors = [bar_color if c else "white" for c in cleared]

    bars = ax.bar(ep_idx, pf_rec, color=colors, edgecolor="black",
                  linewidth=1.0, width=0.7, zorder=3)
    for bar, h in zip(bars, [None if c else hatch_uncleared for c in cleared]):
        if h:
            bar.set_hatch(h)

    ax.set_title(f"RECALL AT FLOOR ($\geq${int(precision_floor*100)}%)", pad=4)
    ax.set_xlabel("Epoch", labelpad=2)
    ax.set_ylabel("Recall (%)", labelpad=2)
    ax.set_xlim(min(ep_idx) - 0.5, max(ep_idx) + 0.5)
    ax.set_ylim(0, 105)

    legend_elems = [
        Patch(facecolor=bar_color, edgecolor="black", label="Cleared"),
        Patch(facecolor="white", edgecolor="black", hatch=hatch_uncleared, label="Uncleared"),
    ]
    leg = ax.legend(handles=legend_elems, loc="upper left", handlelength=1.4)
    leg.get_frame().set_linewidth(1.0)
    style_axes(ax)
    savefig(fig, out_dir, "fig2_precision_floor")


# --------------------------------------------------------------------------
# Figure 3: Metrics vs Discharge Rate Sweep (Alpha Tradeoff)
# --------------------------------------------------------------------------

def plot_alpha_tradeoff(d: dict, out_dir: str) -> None:
    rows = d.get("power_latency_tradeoff", [])
    if not rows:
        print("[!] Warning: 'power_latency_tradeoff' missing in metrics.json. Skipping Fig 3.")
        return

    alpha = [r["alpha"] for r in rows]
    f1 = [r["f1"] * 100 for r in rows]
    rec = [r["recall"] * 100 for r in rows]
    prec = [r["precision"] * 100 for r in rows]
    acc = [r["accuracy"] * 100 for r in rows]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    for key, values, label in [
        ("f1", f1, "F1"), ("recall", rec, "Recall"),
        ("precision", prec, "Prec."), ("accuracy", acc, "Acc."),
    ]:
        s = SERIES_STYLES[key]
        ax.plot(alpha, values, color=s["color"], marker=s["marker"], ls=s["ls"],
                lw=1.3, mfc="none", mew=1.0, ms=4.5, label=label)

    ax.set_title("ROBUSTNESS-BUDGET SWEEP", pad=4)
    ax.set_xlabel(r"Discharge Rate ($\alpha$)", labelpad=2)
    ax.set_ylabel("Metric Score (%)", labelpad=2)
    ax.set_xlim(min(alpha), max(alpha))
    ax.set_ylim(0, 105)

    leg = ax.legend(loc="lower left", ncol=2, handlelength=1.6)
    leg.get_frame().set_linewidth(1.0)
    style_axes(ax)
    savefig(fig, out_dir, "fig3_alpha_tradeoff")


# --------------------------------------------------------------------------
# Figure 4: Halting Latency vs Discharge Rate
# --------------------------------------------------------------------------

def plot_exit_vs_alpha(d: dict, out_dir: str) -> None:
    rows = d.get("power_latency_tradeoff", [])
    if not rows:
        return

    alpha = [r["alpha"] for r in rows]
    t_exit = [r["mean_t_halt"] for r in rows]
    t_max = d.get("config", {}).get("t_max", 100)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.axhline(t_max, color="black", ls=":", lw=1.2, zorder=1, label=r"$T_{max}$ Bound")
    ax.plot(alpha, t_exit, color="#00008B", marker="o", ls="-", lw=1.5,
            mfc="none", mew=1.2, ms=4.5, zorder=3, label=r"$\langle T_{exit} \rangle$")

    ax.set_title("HALTING LATENCY", pad=4)
    ax.set_xlabel(r"Discharge Rate ($\alpha$)", labelpad=2)
    ax.set_ylabel(r"Exit Timestep $\langle T_{exit} \rangle$", labelpad=2)
    ax.set_xlim(min(alpha), max(alpha))
    ax.set_ylim(0, t_max * 1.08)

    leg = ax.legend(loc="upper right", handlelength=1.6)
    leg.get_frame().set_linewidth(1.0)
    style_axes(ax)
    savefig(fig, out_dir, "fig4_exit_vs_alpha")


# --------------------------------------------------------------------------
# Figure 5: Weight Distribution / Sparsity (From Model Weights File)
# --------------------------------------------------------------------------

def plot_weight_distribution(weights_path: str, out_dir: str) -> None:
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[!] Warning: Model weights file '{weights_path}' not found. Skipping Fig 5.")
        return

    if not TORCH_AVAILABLE:
        print("[!] Warning: PyTorch is not installed. Cannot parse model weights. Skipping Fig 5.")
        return

    print(f"[*] Loading model checkpoint: {weights_path}")
    try:
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        
        all_weights = []
        layer_norms = []
        layer_names = []

        for name, param in state_dict.items():
            if "weight" in name and isinstance(param, torch.Tensor) and param.ndim > 1:
                w = param.detach().numpy().flatten()
                all_weights.append(w)
                layer_norms.append(np.linalg.norm(w))
                # Truncate layer name for plotting clarity
                short_name = name.replace(".weight", "").split(".")[-1]
                layer_names.append(short_name)

        if not all_weights:
            print("[!] No multi-dimensional weight tensors found in checkpoint.")
            return

        flat_all = np.concatenate(all_weights)

        # Plot 5a: Overall Weight Magnitude Histogram
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.hist(flat_all, bins=60, color="#1B4F72", edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(0, color="red", ls="--", lw=1.0)
        ax.set_title("WEIGHT DISTRIBUTION", pad=4)
        ax.set_xlabel("Weight Value", labelpad=2)
        ax.set_ylabel("Count", labelpad=2)
        style_axes(ax)
        savefig(fig, out_dir, "fig5a_weight_histogram")

        # Plot 5b: Layer-wise Weight Norms
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        x_idx = np.arange(len(layer_norms))
        ax.bar(x_idx, layer_norms, color="#2E86C1", edgecolor="black", linewidth=0.8, width=0.6)
        ax.set_xticks(x_idx)
        ax.set_xticklabels(layer_names, rotation=45, ha="right", fontsize=7.0)
        ax.set_title("LAYER-WISE L2 NORMS", pad=4)
        ax.set_xlabel("Layer", labelpad=2)
        ax.set_ylabel(r"$\|W\|_2$ Norm", labelpad=2)
        style_axes(ax)
        savefig(fig, out_dir, "fig5b_layer_norms")

    except Exception as e:
        print(f"[!] Failed to parse weights file: {e}")


# --------------------------------------------------------------------------
# Figure 6: Jagged Hardware Voltage Trace (Accepted Reference Match)
# --------------------------------------------------------------------------

def plot_voltage_simulation(d: dict, out_dir: str) -> None:
    rows = {r["alpha"]: r["mean_t_halt"] for r in d.get("power_latency_tradeoff", [])}
    alphas = sorted(rows.keys()) if rows else [0.003, 0.006, 0.010, 0.014, 0.020, 0.028]
    t_max = d.get("config", {}).get("t_max", 100)
    v_critical = d.get("config", {}).get("v_critical", 0.3)
    v_emergency = d.get("config", {}).get("v_emergency", 0.05)
    
    t = np.arange(t_max, dtype=np.float64)
    rng = np.random.RandomState(42)

    styles = [
        {"color": "#111111", "ls": "-"},
        {"color": "#222222", "ls": (0, (6, 3))},
        {"color": "#333333", "ls": (0, (4, 2, 1, 2))},
        {"color": "#444444", "ls": (0, (1, 1.5))},
        {"color": "#666666", "ls": (0, (3, 1, 1, 1))},
        {"color": "#888888", "ls": (0, (5, 1, 1, 1, 1, 1))}
    ]

    fig, ax = plt.subplots(figsize=(3.6, 2.2))

    for i, alpha in enumerate(alphas):
        style = styles[i % len(styles)]
        v_base = 1.0 - alpha * t
        noise = rng.normal(0.0, 0.010, size=t_max)
        v_dd = np.clip(v_base + noise, 0.0, 1.0)

        ax.plot(t, v_dd, color=style["color"], linestyle=style["ls"],
                linewidth=1.1, label=fr"$\alpha={alpha:.3f}$")

        t_exit = rows.get(alpha)
        if t_exit is not None and t_exit < t_max:
            t_exit_int = int(round(t_exit))
            v_at_exit = v_dd[min(t_exit_int, t_max - 1)]
            ax.plot(t_exit, v_at_exit, marker="o", markersize=5.5,
                    mfc="white", mec="#444444", mew=1.4, zorder=5)

    ax.axhline(v_critical, color="#c0392b", linestyle="--", linewidth=1.2, label=r"$V_{crit}$")
    ax.axhline(v_emergency, color="#c0392b", linestyle=":", linewidth=1.2, label=r"$V_{emerg}$")

    ax.set_xlabel(r"Timestep $t$", labelpad=3)
    ax.set_ylabel(r"Simulated supply voltage $V_{dd}(t)$", labelpad=3)
    ax.set_xlim(0, t_max)
    ax.set_ylim(-0.02, 1.05)

    leg = ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                    frameon=True, fontsize=7.5, handlelength=2.2)
    leg.get_frame().set_edgecolor("black")
    leg.get_frame().set_linewidth(1.0)

    style_axes(ax)
    savefig(fig, out_dir, "fig6_voltage_simulation")


# --------------------------------------------------------------------------
# Main Orchestrator
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate full IEEE benchmark figures.")
    parser.add_argument("--metrics_json", type=str, required=True,
                        help="Path to metrics.json file")
    parser.add_argument("--weights_file", type=str, default=None,
                        help="Optional path to PyTorch model checkpoint (.pt/.pth)")
    parser.add_argument("--out_dir", type=str, default="figures",
                        help="Output directory for generated plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    d = load_metrics(args.metrics_json)

    print(f"[*] Generating plots in directory: {args.out_dir}")
    plot_training_curves(d, args.out_dir)
    plot_precision_floor(d, args.out_dir)
    plot_alpha_tradeoff(d, args.out_dir)
    plot_exit_vs_alpha(d, args.out_dir)
    plot_voltage_simulation(d, args.out_dir)

    if args.weights_file:
        plot_weight_distribution(args.weights_file, args.out_dir)

    print(f"\n[+] All plots successfully exported to: {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()