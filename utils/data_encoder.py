import os
import wfdb
import torch
import numpy as np
import matplotlib.pyplot as plt

def download_mit_bih(download_dir="../data/raw"):
    """Downloads a sample of the MIT-BIH Arrhythmia Database if not present."""
    os.makedirs(download_dir, exist_ok=True)
    record_name = '100' # Record 100 is a standard baseline with some arrhythmias
    
    if not os.path.exists(os.path.join(download_dir, f"{record_name}.dat")):
        print(f"[*] Downloading MIT-BIH Record {record_name}...")
        wfdb.dl_database('mitdb', download_dir, records=[record_name])
    
    return os.path.join(download_dir, record_name)

def delta_modulation_encoder(signal, threshold):
    """
    Translates a continuous analog waveform into a sparse binary spike train.
    If the voltage changes by more than the threshold, it fires a spike.
    """
    spikes = np.zeros_like(signal)
    reference = signal[0]
    
    for t in range(1, len(signal)):
        diff = signal[t] - reference
        if diff >= threshold:
            spikes[t] = 1        # Positive spike
            reference = signal[t]
        elif diff <= -threshold:
            spikes[t] = -1       # Negative spike
            reference = signal[t]
            
    return spikes

def process_and_visualize():
    record_path = download_mit_bih()
    
    # Load 3 seconds of the ECG recording (Sampling rate is 360 Hz)
    print("[*] Loading ECG Data...")
    record = wfdb.rdrecord(record_path, sampto=1080)
    ecg_signal = record.p_signal[:, 0] # Lead MLII
    
    # Encode the signal (Adjust threshold to tune sparsity)
    print("[*] Delta Modulating Signal...")
    spike_train = delta_modulation_encoder(ecg_signal, threshold=0.15)
    
    # --- Visualization for your Paper ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    ax1.plot(ecg_signal, color='blue')
    ax1.set_title("Raw MIT-BIH ECG Waveform (Analog)")
    ax1.set_ylabel("Voltage (mV)")
    
    # Plot positive spikes in black, negative spikes in red
    pos_spikes = np.where(spike_train == 1)[0]
    neg_spikes = np.where(spike_train == -1)[0]
    
    ax2.vlines(pos_spikes, ymin=0, ymax=1, colors='black', label="Positive Spike")
    ax2.vlines(neg_spikes, ymin=-1, ymax=0, colors='red', label="Negative Spike")
    ax2.set_title("Delta-Modulated Spike Train (Neuromorphic Input)")
    ax2.set_xlabel("Time Steps (360 Hz)")
    ax2.set_ylabel("Spike Amplitude")
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig('../plots/ecg_spike_encoding.png', dpi=300)
    print("[+] Processing complete. Encoding graph saved to plots/ecg_spike_encoding.png")

if __name__ == "__main__":
    process_and_visualize()