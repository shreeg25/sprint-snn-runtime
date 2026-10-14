import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, WeightedRandomSampler

# Import your established architecture
from train_snn import MITBIHSpikeDataset, WearableSNN

def log_status(msg):
    sys.stdout.write(f"[HARDWARE SIM] {msg}\n")
    sys.stdout.flush()

def generate_co_design_plot(df):
    log_status("Rendering IEEE Power-Aware Trade-off Graphic...")
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 11})
    
    fig, ax1 = plt.subplots(figsize=(11, 7))
    
    # X-axis
    budgets = df['Initial_Budget_Category'].values
    recall = df['Recall'].values
    snn_energy = df['Proposed_SNN_Energy_nJ'].values
    cnn_nvm_energy = df['Baseline_NVM_Energy_nJ'].values
    x = np.arange(len(budgets))
    
    # Left Axis (Recall)
    color = '#d62728'
    ax1.set_xlabel('AVAILABLE CAPACITOR ENERGY BUDGET (High to Low)', fontweight='bold')
    ax1.set_ylabel('SNN CLINICAL RECALL (%)', color=color, fontweight='bold')
    ax1.plot(x, recall, color=color, marker='o', linewidth=3, markersize=10, label='Proposed SNN Recall')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 105])
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right Axis (Energy - Log Scale)
    ax2 = ax1.twinx()
    color2 = '#1f77b4'
    color3 = '#7f7f7f'
    ax2.set_ylabel('SYSTEM ENERGY COST (nJ) [LOG SCALE]', color='black', fontweight='bold')
    
    width = 0.3
    ax2.bar(x - width/2, snn_energy, width, color=color2, edgecolor='black', alpha=0.8, label='Proposed SNN (No NVM)')
    ax2.bar(x + width/2, cnn_nvm_energy, width, color=color3, edgecolor='black', alpha=0.8, label='Baseline + NVM Write')
    
    # Fix Annotation: Use multiplication for Log Scale offset
    spike_val = cnn_nvm_energy[3]
    ax2.annotate('Catastrophic NVM\nEnergy Spike', 
                 xy=(3 + width/2, spike_val), xytext=(2.8, spike_val * 4.0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 fontweight='bold', ha='right')

    ax2.set_yscale('log')
    # Dynamically scale the top boundary so the annotation fits inside the plot
    ax2.set_ylim([10, spike_val * 30]) 
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(budgets, fontweight='bold')
    
    # Fix Legend Overlap
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    
    # Add a padded title to make room for the legend underneath
    plt.title('POWER-AWARE ELASTICITY: RECALL VS HARDWARE SURVIVAL', fontweight='bold', pad=45)
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, framealpha=1.0, edgecolor='black')

    # The crucial fix: bbox_inches='tight' prevents Matplotlib from cropping outside text
    plot_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, 'power_aware_hardware_tradeoff.pdf')
    plt.savefig(out_path, format='pdf', dpi=300, bbox_inches='tight')
    log_status(f"Saved graphic to {out_path}")
    plt.close()

if __name__ == "__main__":
    device = torch.device("cpu")
    log_status("Initializing Global Dataset...")
    
    dataset = MITBIHSpikeDataset(data_dir='data/raw', window_size=100, threshold=0.15)
    class_counts = np.bincount(dataset.labels)
    sampler = WeightedRandomSampler(weights=[1.0 / class_counts[label] for label in dataset.labels], num_samples=2000, replacement=True)
    dataloader = DataLoader(dataset, batch_size=2000, sampler=sampler)

    log_status("Loading 20-Epoch SNN Weights...")
    snn_model = WearableSNN(num_hidden=256).to(device)
    snn_model.load_state_dict(torch.load('models/distilled_snn_weights.pt', map_location=device, weights_only=True))
    snn_model.eval()

    # Get a massive batch of data to simulate
    data, targets = next(iter(dataloader))
    data, targets = data.transpose(0, 1).to(device), targets.to(device)

    # Hardware Constants (in nanoJoules for scaling)
    E_SOP_nJ = 0.0009       # Energy per SNN spike accumulate
    E_TX_nJ = 50.0          # Energy to fire the 6G Wake-up Radio
    E_NVM_WRITE_nJ = 1500.0 # Energy to write baseline state to ReRAM (Catastrophic)
    
    # Simulate different initial battery states (from 100% full down to 25% critical)
    battery_profiles = [
        {'name': '100% (Abundant)', 'decay_rate': 0.0, 'base_threshold': 5.0},
        {'name': '75% (Nominal)', 'decay_rate': 0.02, 'base_threshold': 4.0},
        {'name': '50% (Low)', 'decay_rate': 0.05, 'base_threshold': 2.5},
        {'name': '25% (Critical)', 'decay_rate': 0.15, 'base_threshold': 1.0}
    ]

    results = []

    with torch.no_grad():
        _, mem_rec = snn_model(data) 
        # Extract the continuous voltage confidence: |V_arrhythmia - V_normal|
        confidence_trajectory = torch.abs(mem_rec[:, :, 1] - mem_rec[:, :, 0]).cpu()

        for profile in battery_profiles:
            log_status(f"Simulating Power State: {profile['name']}")
            
            t_halts = []
            preds = []
            snn_energy_total = 0.0
            baseline_energy_total = 0.0
            
            for b in range(confidence_trajectory.size(1)): # Over each heartbeat
                forced_exit_t = 100
                pred_class = 0
                
                for t in range(100):
                    # SUPPLY-MODULATED THRESHOLD: The barrier drops as battery decays
                    current_threshold = max(0.1, profile['base_threshold'] - (profile['decay_rate'] * t))
                    
                    if confidence_trajectory[t, b] >= current_threshold:
                        forced_exit_t = t + 1
                        pred_class = 1 if mem_rec[t, b, 1] > mem_rec[t, b, 0] else 0
                        break
                
                # If it never crossed, take the final timestep prediction
                if forced_exit_t == 100:
                    pred_class = 1 if mem_rec[-1, b, 1] > mem_rec[-1, b, 0] else 0

                t_halts.append(forced_exit_t)
                preds.append(pred_class)
                
                # ENERGY CALCULATION
                # Proposed: SOP energy up to exit + Radio Wake-up (No NVM)
                snn_energy_total += (forced_exit_t * 256 * E_SOP_nJ) + E_TX_nJ
                
                # Baseline: If it is forced to stop early due to power, it MUST write to NVM
                if forced_exit_t < 60 and profile['name'] in ['50% (Low)', '25% (Critical)']:
                    baseline_energy_total += E_NVM_WRITE_nJ
                else:
                    baseline_energy_total += (100 * 256 * 0.0046) # Continuous MAC cost
                    
            recall = recall_score(targets.cpu().numpy(), preds, zero_division=0) * 100
            
            results.append({
                'Initial_Budget_Category': profile['name'],
                'Average_T_Halt': np.mean(t_halts),
                'Recall': recall,
                'Proposed_SNN_Energy_nJ': snn_energy_total / len(preds),
                'Baseline_NVM_Energy_nJ': baseline_energy_total / len(preds)
            })

    df = pd.DataFrame(results)
    print("\n=======================================================")
    print(df.to_string(index=False))
    print("=======================================================\n")
    
    generate_co_design_plot(df)