import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score, precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import the architecture and dataset strictly from your source
from train_snn import MITBIHSpikeDataset, WearableSNN

def log_status(msg):
    sys.stdout.write(f"[ELASTICITY] {msg}\n")
    sys.stdout.flush()

def generate_elasticity_plot(df):
    log_status("Rendering IEEE-compliant Elasticity Curve...")
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 12})
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    timesteps = df['Timesteps'].values
    recall = df['Recall'].values
    precision = df['Precision'].values

    # Plot the SNN's dynamic temporal evolution
    ax.plot(timesteps, recall, color='#d62728', marker='o', linewidth=3, markersize=8, label='SNN Dynamic Recall')
    
    # 75% Clinical Wake-Up Threshold
    ax.axhline(y=75.0, color='#333333', linestyle=':', linewidth=2, label='Clinical Wake-Up Threshold (75%)')
    
    # Annotate the Early Exit point
    early_exit_idx = np.where(recall >= 75.0)[0]
    if len(early_exit_idx) > 0:
        exit_t = timesteps[early_exit_idx[0]]
        exit_r = recall[early_exit_idx[0]]
        ax.plot(exit_t, exit_r, marker='s', markersize=12, color='black')
        ax.annotate(f'Early Exit Trigger\n(T={exit_t}, Recall={exit_r:.1f}%)', 
                    xy=(exit_t, exit_r), xytext=(exit_t + 5, exit_r - 15),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                    fontweight='bold')

    # Shade the "Inelastic Dead Zone" for continuous models
    ax.axvspan(0, 99, color='gray', alpha=0.15, label='CNN/RF Inelastic Dead Zone')
    ax.axvline(x=100, color='#1f77b4', linestyle='--', linewidth=3, label='CNN/RF Minimum Latency (T=100)')

    ax.set_xlim([10, 105])
    ax.set_ylim([0, 105])
    ax.set_xlabel('TIMESTEPS PROCESSED (Latency)')
    ax.set_ylabel('METRIC PERCENTAGE (%)')
    ax.set_title('ELASTIC RUNTIME VS. STATIC BASELINES')
    ax.legend(loc='lower right', framealpha=1.0, edgecolor='black')
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, 'snn_elasticity_curve.pdf')
    plt.savefig(out_path, format='pdf', dpi=300)
    log_status(f"Saved graphic to {out_path}")

if __name__ == "__main__":
    device = torch.device("cpu")
    log_status("Initializing Global Dataset...")
    
    dataset = MITBIHSpikeDataset(data_dir='data/raw', window_size=100, threshold=0.15)
    class_counts = np.bincount(dataset.labels)
    sample_weights = [1.0 / class_counts[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=256, sampler=sampler)

    log_status("Loading 20-Epoch Proposed SNN Weights...")
    snn_model = WearableSNN(num_hidden=256).to(device)
    weight_path = 'models/distilled_snn_weights.pt'
    snn_model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    snn_model.eval()

    temporal_checkpoints = range(10, 101, 10)
    results = []

    with torch.no_grad():
        for T in temporal_checkpoints:
            log_status(f"Evaluating SNN Elastic Early-Exit at T={T}...")
            t_all, p_all = [], []
            
            for data, targets in dataloader:
                # Transpose to (Time, Batch, Features)
                data = data.transpose(0, 1).to(device)
                
                # ARTIFICIAL TRUNCATION: Force the SNN to decide using only early spikes
                data_truncated = data[:T]
                
                _, mem_rec = snn_model(data_truncated)
                _, preds = mem_rec.sum(dim=0).max(1)
                
                t_all.extend(targets.cpu().numpy())
                p_all.extend(preds.cpu().numpy())
                
            recall = recall_score(t_all, p_all, zero_division=0) * 100
            precision = precision_score(t_all, p_all, zero_division=0) * 100
            results.append({'Timesteps': T, 'Recall': recall, 'Precision': precision})

    df = pd.DataFrame(results)
    
    txt_path = os.path.join(os.getcwd(), 'elasticity_metrics.txt')
    with open(txt_path, 'w') as f:
        f.write("ELASTIC RUNTIME EVOLUTION\n")
        f.write("=========================\n")
        for index, row in df.iterrows():
            f.write(f"T={int(row['Timesteps']):03d} | Recall: {row['Recall']:5.2f}% | Precision: {row['Precision']:5.2f}%\n")
    
    generate_elasticity_plot(df)