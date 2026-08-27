import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# --- IEEE Publication Styling Overrides ---
plt.rcParams.update({
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.linewidth': 2,
    'lines.linewidth': 2,
    'font.family': 'sans-serif',
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.color': 'black',
    'axes.labelcolor': 'black'
})

def log_status(msg):
    sys.stdout.write(f"[RUNNING] {msg}\n")
    sys.stdout.flush()

def generate_synthetic_ecg(duration=3.0, fs=360):
    """Mathematically synthesizes a human heartbeat (P-QRS-T waves)"""
    log_status("Synthesizing human ECG waveform via Gaussian math...")
    t = np.linspace(0, duration, int(duration * fs))
    ecg = np.zeros_like(t)
    
    # 60 BPM Heartbeat generator
    for beat in range(int(duration)):
        ecg += 0.15 * np.exp(-((t - (beat + 0.2)) ** 2) / (2 * 0.02 ** 2))    # P-wave
        ecg -= 0.10 * np.exp(-((t - (beat + 0.4)) ** 2) / (2 * 0.01 ** 2))    # Q-wave
        ecg += 1.20 * np.exp(-((t - (beat + 0.45)) ** 2) / (2 * 0.015 ** 2))  # R-wave (peak)
        ecg -= 0.20 * np.exp(-((t - (beat + 0.5)) ** 2) / (2 * 0.01 ** 2))    # S-wave
        ecg += 0.30 * np.exp(-((t - (beat + 0.7)) ** 2) / (2 * 0.04 ** 2))    # T-wave
        
    # Add biological noise and baseline wander
    ecg += 0.05 * np.sin(2 * np.pi * 0.5 * t) + np.random.normal(0, 0.015, len(t))
    return ecg

def delta_modulation_encoder(signal, threshold):
    log_status("Executing Delta Modulation encoding algorithm...")
    spikes = np.zeros_like(signal)
    reference = signal[0]
    for t in range(1, len(signal)):
        diff = signal[t] - reference
        if diff >= threshold:
            spikes[t] = 1        
            reference = signal[t]
        elif diff <= -threshold:
            spikes[t] = -1       
            reference = signal[t]
    return spikes

def process_and_visualize():
    # Generate data locally in memory
    ecg_signal = generate_synthetic_ecg()
    spike_train = delta_modulation_encoder(ecg_signal, threshold=0.15)
    
    log_status("Rendering IEEE-compliant vector graphic...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Top Subplot
    ax1.plot(ecg_signal, color='black')
    ax1.set_title("SYNTHETIC ECG WAVEFORM (ANALOG BASELINE)")
    ax1.set_ylabel("VOLTAGE (mV)")
    ax1.grid(True, linestyle='--', alpha=0.5, color='black')
    
    # Bottom Subplot
    pos_spikes = np.where(spike_train == 1)[0]
    neg_spikes = np.where(spike_train == -1)[0]
    
    ax2.vlines(pos_spikes, ymin=0, ymax=1, colors='black', label="POSITIVE SPIKE")
    ax2.vlines(neg_spikes, ymin=-1, ymax=0, colors='black', linestyles='dashed', label="NEGATIVE SPIKE")
    ax2.set_title("DELTA-MODULATED SPIKE TRAIN (NEUROMORPHIC INPUT)")
    ax2.set_xlabel("TIME STEPS (360 Hz)")
    ax2.set_ylabel("SPIKE AMPLITUDE")
    ax2.legend(loc="upper right", frameon=True, edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5, color='black')
    
    plt.tight_layout()
    
    # Save Output
    os.makedirs("../plots", exist_ok=True)
    output_path = "../plots/ecg_spike_encoding.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    log_status(f"SUCCESS: Vector graphic exported to {output_path}")

if __name__ == "__main__":
    process_and_visualize()