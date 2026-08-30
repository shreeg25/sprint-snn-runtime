import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
import wfdb
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score

def log_status(msg):
    sys.stdout.write(f"[RUNNING] {msg}\n")
    sys.stdout.flush()

# --- 1. SNN Architecture (High-Retention) ---
class WearableSNN(nn.Module):
    def __init__(self, num_inputs=1, num_hidden=128, num_outputs=2):
        super(WearableSNN, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        # CHANGED: beta=0.99 to prevent temporal amnesia
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.15, spike_grad=spike_grad) 
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # CHANGED: beta=0.99 to prevent temporal amnesia
        self.lif2 = snn.Leaky(beta=0.9, threshold=0.15, spike_grad=spike_grad) 
        
        nn.init.xavier_uniform_(self.fc1.weight, gain=2.0)
        nn.init.xavier_uniform_(self.fc2.weight, gain=2.0)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_rec, mem2_rec = [], []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step]) * 2.0 
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1) * 2.0
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# --- 2. Asymmetric Temporal Distillation Loss ---
class AsymmetricTemporalDistillationLoss(nn.Module):
    def __init__(self):
        super(AsymmetricTemporalDistillationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma = 0.0 

    def forward(self, spike_trains, targets):
        T, B, C = spike_trains.shape
        spike_count = spike_trains.sum(dim=0) 
        loss_ce = self.ce_loss(spike_count, targets)
        
        # Create a time vector mapping [1 to T]
        time_vector = torch.arange(1, T + 1, dtype=torch.float32, device=spike_trains.device).view(T, 1, 1) 
        
        # Calculate the temporal penalty for EVERY sample in the batch individually
        sample_penalties = torch.sum(spike_trains * time_vector, dim=(0, 2)) # Shape: [Batch_Size]
        
        # ASYMMETRIC MASKING: 
        # Only penalize late spikes if the ground truth is Normal (Class 0).
        # We give the network a free pass to investigate Arrhythmias (Class 1) to protect Recall.
        normal_mask = (targets == 0).float()
        
        # Avoid division by zero if a batch somehow has no Normal beats
        valid_normal_beats = normal_mask.sum() + 1e-8
        
        late_spike_penalty = torch.sum(sample_penalties * normal_mask) / valid_normal_beats
        
        return loss_ce + (self.gamma * late_spike_penalty), loss_ce, late_spike_penalty
    
import glob

# --- 3. Global Biological Dataloader (Multi-Patient MIT-BIH) ---
class MITBIHSpikeDataset(Dataset):
    def __init__(self, data_dir='data/raw', window_size=100, threshold=0.15):
        log_status(f"Scanning {data_dir} for multi-patient biological records...")
        
        # Find all .dat files in the directory and strip the extension
        record_paths = [f.replace('.dat', '') for f in glob.glob(f"{data_dir}/*.dat")]
        
        if not record_paths:
            raise FileNotFoundError(f"CRITICAL FAULT: No .dat files found in {data_dir}.")
            
        self.signal = []
        self.peaks = []
        self.symbols = []
        
        current_offset = 0
        
        # Sequentially ingest every patient in the database
        for path in record_paths:
            try:
                record = wfdb.rdrecord(path)
                annotation = wfdb.rdann(path, 'atr')
                
                sig = record.p_signal[:, 0]
                
                # Shift the cardiologist's manual peak indices by the length of previous signals
                shifted_peaks = annotation.sample + current_offset
                
                self.signal.append(sig)
                self.peaks.append(shifted_peaks)
                self.symbols.append(np.array(annotation.symbol))
                
                current_offset += len(sig)
            except Exception as e:
                log_status(f"WARNING: Skipping {path} due to parsing error: {e}")
            
        # Concatenate all patients into massive contiguous global arrays
        self.signal = np.concatenate(self.signal)
        self.peaks = np.concatenate(self.peaks)
        self.symbols = np.concatenate(self.symbols)
        
        # Filter for valid beats across the entire population
        valid_beats = np.isin(self.symbols, ['N', 'V', 'A', 'L', 'R'])
        self.peaks = self.peaks[valid_beats]
        self.symbols = self.symbols[valid_beats]
        
        self.window_size = window_size
        self.threshold = threshold
        
        # Expose labels for the WeightedRandomSampler to balance
        self.labels = [0 if sym == 'N' else 1 for sym in self.symbols]
        
        log_status(f"Global Dataset Ready: Isolated {len(self.peaks)} heartbeats across {len(record_paths)} patients.")

    def _delta_modulate(self, segment):
        spikes = np.zeros_like(segment)
        reference = segment[0]
        for t in range(1, len(segment)):
            diff = segment[t] - reference
            if diff >= self.threshold:
                spikes[t] = 1; reference = segment[t]
            elif diff <= -self.threshold:
                spikes[t] = -1; reference = segment[t]
        return spikes

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        peak_idx = self.peaks[idx]
        start = max(0, peak_idx - self.window_size // 2)
        end = start + self.window_size
        
        segment = self.signal[start:end]
        if len(segment) < self.window_size:
            segment = np.pad(segment, (0, self.window_size - len(segment)), 'constant')
            
        spike_train = self._delta_modulate(segment)
        label = self.labels[idx]
        
        return torch.tensor(spike_train, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.peaks)

    def __getitem__(self, idx):
        peak_idx = self.peaks[idx]
        start = max(0, peak_idx - self.window_size // 2)
        end = start + self.window_size
        
        segment = self.signal[start:end]
        if len(segment) < self.window_size:
            segment = np.pad(segment, (0, self.window_size - len(segment)), 'constant')
            
        spike_train = self._delta_modulate(segment)
        label = 0 if self.symbols[idx] == 'N' else 1
        
        return torch.tensor(spike_train, dtype=torch.float32).unsqueeze(-1), torch.tensor(label, dtype=torch.long)

# --- 4. Execution Engine (Exponential Curriculum) ---
if __name__ == "__main__":
    device = torch.device("cpu")
    log_status("Execution Engine: CPU")
    
    TIME_STEPS = 100
    EPOCHS = 20 # Increased to 20 to allow the exponential curve to bite
    
    model = WearableSNN(num_hidden=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    
    # Pass the directory, not a hardcoded single file
    # Hypersensitize the edge-encoder to capture low-frequency pathologies
    dataset = MITBIHSpikeDataset(data_dir='data/raw', window_size=TIME_STEPS, threshold=0.15)
    
    # --- The Oversampling Architecture ---
    log_status("Calculating global biological class distribution...")
    class_counts = np.bincount(dataset.labels)
    log_status(f"Global Distribution -> Normal: {class_counts[0]}, Arrhythmia: {class_counts[1]}")
    
    # 1. Calculate the physical sample weights for the Sampler
    class_weights_np = 1.0 / class_counts
    sample_weights = [class_weights_np[label] for label in dataset.labels]
    
    # 2. Initialize the standard, unweighted loss function (batches are now 50/50)
    criterion = AsymmetricTemporalDistillationLoss() 
    
    # 3. Force the dataloader to oversample the anomalies
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    for epoch in range(EPOCHS):
        # The Exponential Scheduler
        if epoch < 5:
            criterion.gamma = 0.0  # Phase 1: Pure Accuracy Warmup
        else:
            # Phase 2: Exponential Penalty Ramp-Up
            criterion.gamma = 1e-5 * (1.5 ** (epoch - 5)) 
        
        total_loss, total_penalty = 0, 0
        all_targets = []
        all_preds = []
        model.train()
        
        for data, targets in dataloader:
            data, targets = data.transpose(0, 1).to(device), targets.to(device)
            spk_rec, _ = model(data)
            
            loss, ce, late_penalty = criterion(spk_rec, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_penalty += late_penalty.item()
            
# --- MEDICAL INFERENCE LOGIC (DYNAMIC THRESHOLD) ---
            # 1. Get raw spike counts for Class 0 (Normal) and Class 1 (Arrhythmia)
            spike_counts = spk_rec.sum(dim=0) 
            normal_spikes = spike_counts[:, 0]
            arrhythmia_spikes = spike_counts[:, 1]
            
            # 2. THE THRESHOLD PARAMETER (Alpha)
            # THE THRESHOLD PARAMETER (Alpha)
            alpha = 0.80 
            predicted = (arrhythmia_spikes > (normal_spikes * alpha)).long()
            
            # 4. Store for rigorous medical metrics
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            
        # Calculate Medical Classification Metrics
        acc = 100. * np.mean(np.array(all_preds) == np.array(all_targets))
        recall = recall_score(all_targets, all_preds, zero_division=0) * 100.
        precision = precision_score(all_targets, all_preds, zero_division=0) * 100.
        f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Gamma: {criterion.gamma:.6f} | Loss: {total_loss:.4f} | Penalty: {total_penalty:.2f}")
        print(f"          -> Acc: {acc:.1f}% | Recall: {recall:.1f}% | Precision: {precision:.1f}% | F1: {f1:.1f}%")
        
    os.makedirs("models", exist_ok=True)
    save_path = 'models/distilled_snn_weights.pt'
    torch.save(model.state_dict(), save_path)
    log_status(f"SUCCESS: Hardware SNN weights extracted to {save_path}")