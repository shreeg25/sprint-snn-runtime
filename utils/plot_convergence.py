import matplotlib.pyplot as plt
import numpy as np
import os

# --- IEEE Publication Styling Overrides ---
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'lines.linewidth': 3,
    'font.family': 'sans-serif',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.color': 'black',
    'axes.labelcolor': 'black'
})

def generate_convergence_plot():
    epochs = np.arange(1, 21)
    
    # Exact biological run data from the 20-epoch Curriculum Learning output
    accuracy = [1.5, 82.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 
                98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5, 98.5]
    
    penalty = [365640, 688208, 725476, 725545, 725653, 724783, 723598, 724849, 724981, 723472, 
               706195, 376314, 363590, 363585, 363585, 363585, 363585, 363583, 363580, 202761]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Subplot 1: Accuracy ---
    ax1.plot(epochs, accuracy, color='black', marker='o', markersize=8)
    ax1.set_ylabel('ACCURACY (%)')
    ax1.set_title('CURRICULUM LEARNING: CLASSIFICATION ACCURACY')
    ax1.grid(True, linestyle='--', alpha=0.5, color='black')
    ax1.set_ylim(0, 105)
    
    # Phase 2 Demarcation
    ax1.axvline(x=5.5, color='black', linestyle=':', linewidth=2, label='PHASE 2 (PRUNING) START')
    ax1.legend(loc="lower right", frameon=True, edgecolor='black')

    # --- Subplot 2: Penalty ---
    ax2.plot(epochs, penalty, color='black', marker='s', markersize=8, linestyle='-')
    ax2.set_xlabel('TRAINING EPOCHS')
    ax2.set_ylabel('LATE SPIKE PENALTY')
    ax2.set_title('SPATIOTEMPORAL PRUNING: PENALTY COLLAPSE')
    ax2.grid(True, linestyle='--', alpha=0.5, color='black')
    ax2.axvline(x=5.5, color='black', linestyle=':', linewidth=2)

    # Annotations for the mathematical collapses
    ax2.annotate('First Prune', xy=(12, 376314), xytext=(9, 200000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 horizontalalignment='center', weight='bold')

    ax2.annotate('Deep Prune', xy=(20, 202761), xytext=(17, 400000),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                 horizontalalignment='center', weight='bold')

    plt.tight_layout()
    
    # Save the output
    os.makedirs("../plots", exist_ok=True)
    output_path = '../plots/snn_convergence_metrics.pdf'
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] High-resolution convergence plot saved to {output_path}")

if __name__ == "__main__":
    generate_convergence_plot()