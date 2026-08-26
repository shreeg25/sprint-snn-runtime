import os
import sys
import time
import wfdb
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
    """Provides explicit terminal feedback without external dependencies."""
    sys.stdout.write(f"[RUNNING] {msg}\n")
    sys.stdout.flush()

def download_mit_bih(download_dir="../data/raw"):
    os.makedirs(download_dir, exist_ok=True)
    record_name = '100'
    record_path = os.path.join(download_dir, record_name)
    
    if not os.path.exists(f"{record_path}.dat"):
        log_status("Contacting PhysioNet servers...")
        log_status("Downloading MIT-BIH Record 100. Please wait, this requires network I/O.")
        wfdb.dl_database('mitdb', download_dir, records=[record_name])
        log_status("Download sequence complete.")
    else:
        log_status("Local MIT-BIH dataset found. Bypassing network download.")
        
    return record_path

def delta_modulation_encoder(signal, threshold):
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
    record_path = download_mit_bih()
    
    log_status("Loading 3 seconds of ECG MLII data into memory...")
    record = wfdb.rdrecord(record_path, sampto=1080)
    ecg_signal = record.p_signal[:, 0]
    
    log_status("Executing Delta Modulation encoding algorithm...")
    t_start = time.time()
    spike_train = delta_modulation_encoder(ecg_signal, threshold=0.15)
    log_status(f"Encoding mathematical pass complete in {time.time() - t_start:.4f}s.")
    
    log_status("Rendering IEEE-compliant vector graphic...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    
    # Top Subplot: Analog Waveform
    ax1.plot(ecg_signal, color='black')
    ax1.set_title("RAW MIT-BIH ECG WAVEFORM (ANALOG)")
    ax1.set_ylabel("VOLTAGE (mV)")
    ax1.grid(True, linestyle='--', alpha=0.5, color='black')
    
    # Bottom Subplot: Spikes
    pos_spikes = np.where(spike_train == 1)[0]
    neg_spikes = np.where(spike_train == -1)[0]
    
    ax2.vlines(pos_spikes, ymin=0, ymax=1, colors='black', label="POSITIVE SPIKE")
    # Using dashed lines for negative spikes for monochromatic contrast in print
    ax2.vlines(neg_spikes, ymin=-1, ymax=0, colors='black', linestyles='dashed', label="NEGATIVE SPIKE")
    ax2.set_title("DELTA-MODULATED SPIKE TRAIN (NEUROMORPHIC INPUT)")
    ax2.set_xlabel("TIME STEPS (360 Hz)")
    ax2.set_ylabel("SPIKE AMPLITUDE")
    ax2.legend(loc="upper right", frameon=True, edgecolor='black')
    ax2.grid(True, linestyle='--', alpha=0.5, color='black')
    
    plt.tight_layout()
    
    # Force vector PDF output with high DPI rasterization fallback
    os.makedirs("../plots", exist_ok=True)
    output_path = "../plots/ecg_spike_encoding.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    log_status(f"SUCCESS: Vector graphic exported to {output_path}")

if __name__ == "__main__":
    process_and_visualize()