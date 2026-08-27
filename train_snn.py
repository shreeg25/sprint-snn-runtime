import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np

def log_status(msg):
    sys.stdout.write(f"[RUNNING] {msg}\n")
    sys.stdout.flush()

# --- 1. SNN Architecture (Hypersensitive LIF) ---
class WearableSNN(nn.Module):
    def __init__(self, num_inputs=1, num_hidden=32, num_outputs=2):
        super(WearableSNN, self).__init__()
        
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        # Dropped threshold to 0.2, increased beta to 0.9 to prevent leak death
        self.lif1 = snn.Leaky(beta=0.9, threshold=0.2, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.9, threshold=0.2, spike_grad=spike_grad)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []

        for step in range(x.size(0)):
            # Amplify the input current to force early activity
            cur1 = self.fc1(x[step]) * 5.0 
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1) * 5.0
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)
    
# --- 2. Temporal Distillation Loss ---
class TemporalDistillationLoss(nn.Module):
    def __init__(self, temporal_penalty_gamma=0.01):
        super(TemporalDistillationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma = temporal_penalty_gamma

    def forward(self, spike_trains, targets):
        T, B, C = spike_trains.shape
        spike_count = spike_trains.sum(dim=0) 
        loss_ce = self.ce_loss(spike_count, targets)
        
        time_vector = torch.arange(1, T + 1, dtype=torch.float32, device=spike_trains.device).view(T, 1, 1) 
        late_spike_penalty = torch.sum(spike_trains * time_vector) / B
        
        return loss_ce + (self.gamma * late_spike_penalty), loss_ce, late_spike_penalty

# --- 3. Zero-Network Learnable Dataset ---
class SyntheticECGDataset(Dataset):
    def __init__(self, num_samples=300, window_size=100, threshold=0.15):
        log_status("Synthesizing learnable mathematical ECG dataset in memory...")
        self.data, self.labels = [], []
        
        for _ in range(num_samples):
            label = np.random.choice([0, 1])
            t = np.linspace(0, 1, window_size)
            signal = np.random.normal(0, 0.02, window_size) # Base biological noise
            
            if label == 0:
                # Normal Beat: Sharp, early peak
                signal += 1.2 * np.exp(-((t - 0.3) ** 2) / (2 * 0.01 ** 2))
            else:
                # Arrhythmia: Inverted, delayed, wide peak
                signal -= 0.8 * np.exp(-((t - 0.6) ** 2) / (2 * 0.04 ** 2))
                
            # Delta Modulate
            spikes = np.zeros_like(signal)
            ref = signal[0]
            for i in range(1, len(signal)):
                diff = signal[i] - ref
                if diff > threshold:
                    spikes[i] = 1; ref = signal[i]
                elif diff < -threshold:
                    spikes[i] = -1; ref = signal[i]
                    
            self.data.append(torch.tensor(spikes, dtype=torch.float32).unsqueeze(-1))
            self.labels.append(torch.tensor(label, dtype=torch.long))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

# --- 4. Execution Engine ---
if __name__ == "__main__":
    device = torch.device("cpu")
    log_status(f"Execution Engine: {device.type.upper()}")

    TIME_STEPS = 100
    EPOCHS = 10 # Increased to 10 to watch the gradual convergence
    
    model = WearableSNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-3)
    # The final mathematical balance
    criterion = TemporalDistillationLoss(temporal_penalty_gamma=0.0001)
    
    dataloader = DataLoader(SyntheticECGDataset(window_size=TIME_STEPS), batch_size=16, shuffle=True)
    log_status("Initiating Temporal Distillation SNN Training on Synthesized Features...")
    
    for epoch in range(EPOCHS):
        total_loss, total_penalty, correct, total = 0, 0, 0, 0
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
            
            # Calculate rough accuracy
            _, predicted = spk_rec.sum(dim=0).max(1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        acc = 100. * correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss:.4f} | Penalty: {total_penalty:.4f} | Accuracy: {acc:.1f}%")

    os.makedirs("models", exist_ok=True)
    save_path = 'models/distilled_snn_weights.pt'
    torch.save(model.state_dict(), save_path)
    log_status(f"SUCCESS: Hardware SNN weights extracted to {save_path}")