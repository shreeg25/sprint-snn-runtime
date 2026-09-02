import os
import sys
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import numpy as np
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score

# Import the dataset exactly as we stabilized it
from train_snn import MITBIHSpikeDataset

def log_status(msg):
    sys.stdout.write(f"[ELASTIC SNN] {msg}\n")
    sys.stdout.flush()

# --- 1. SNN Architecture (3-Neuron Halting Output) ---
class HaltingWearableSNN(nn.Module):
    def __init__(self, num_inputs=1, num_hidden=256, num_outputs=3): # [Normal, Arrhythmia, HALT]
        super(HaltingWearableSNN, self).__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.95, threshold=0.15, spike_grad=spike_grad) 
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.95, threshold=0.15, spike_grad=spike_grad) 
        
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

# --- 2. Differentiable Elastic Loss ---
class ElasticHaltingLoss(nn.Module):
    def __init__(self, device, target_steps=40.0):
        super(ElasticHaltingLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma = 2.0
        self.target_steps = target_steps

    def forward(self, spike_trains, mem_trains, targets, use_mask=True):
        T, B, C = spike_trains.shape
        halt_spikes = spike_trains[:, :, 2] 
        cum_halts = halt_spikes.cumsum(dim=0)
        
        if use_mask:
            active_mask = torch.relu(1.0 - cum_halts + halt_spikes) 
        else:
            active_mask = torch.ones_like(halt_spikes)
            
        t_halt_values = active_mask.sum(dim=0) # Shape: (B,)
        avg_t_halt = t_halt_values.mean()
        
        # Mean Pooling over active timesteps
        class_mem = mem_trains[:, :, :2] 
        masked_mem = class_mem * active_mask.unsqueeze(-1)
        integrated_logits = torch.sum(masked_mem, dim=0) / torch.clamp(t_halt_values.unsqueeze(-1), min=1.0)
        
        loss_ce = self.ce_loss(integrated_logits, targets)
        
        # QUADRATIC TARGET-BOUNDED LATENCY LOSS
        # Penalty applies ONLY when avg_t_halt > target_steps. Drops to 0 below target.
        if use_mask and avg_t_halt > self.target_steps:
            excess_ratio = (avg_t_halt - self.target_steps) / float(T)
            loss_latency = self.gamma * (excess_ratio ** 2)
        else:
            loss_latency = 0.0
            
        total_loss = loss_ce + loss_latency
        
        return total_loss, integrated_logits, avg_t_halt

# --- 3. Execution Engine ---
if __name__ == "__main__":
    device = torch.device("cpu")
    EPOCHS = 20
    
    log_status("Initializing Global Dataset...")
    dataset = MITBIHSpikeDataset(data_dir='data/raw', window_size=100, threshold=0.15)
    class_counts = np.bincount(dataset.labels)
    sample_weights = [1.0 / class_counts[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    model = HaltingWearableSNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = ElasticHaltingLoss(device=device)
    
    # Initialize gamma outside the loop before Epoch 0
    current_gamma = 0.0
    criterion = ElasticHaltingLoss(device, target_steps=40.0)

    for epoch in range(EPOCHS):
        # Epochs 0-4: Learn biological features (Mask OFF)
        # Epochs 5-19: Enforce 40-step latency target (Mask ON)
        if epoch < 5:
            use_mask = False
            criterion.gamma = 0.0  
        else:
            use_mask = True
            criterion.gamma = 2.0  # Constant strong push down to target_steps

        model.train()
        total_loss, total_t_halt = 0, 0
        all_targets, all_preds = [], []

        for data, targets in dataloader:
            data, targets = data.transpose(0, 1).to(device), targets.to(device)
            
            spk_rec, mem_rec = model(data)
            loss, integrated_logits, avg_t_halt = criterion(spk_rec, mem_rec, targets, use_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_t_halt += avg_t_halt.item()
            
            # Clinical Inference at exactly the moment the network halted itself
            _, predicted = integrated_logits.max(1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            # Capture average T_halt at epoch end to guide next epoch's gamma
            avg_t_halt_prev = total_t_halt / len(dataloader)   
        acc = accuracy_score(all_targets, all_preds) * 100.
        rec = recall_score(all_targets, all_preds, zero_division=0) * 100.
        prec = precision_score(all_targets, all_preds, zero_division=0) * 100.
        f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.
        avg_epoch_latency = total_t_halt / len(dataloader)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Gamma: {criterion.gamma:.4f} | Avg T_Halt: {avg_epoch_latency:.1f} steps")
        print(f"          -> Acc: {acc:.1f}% | Recall: {rec:.1f}% | Precision: {prec:.1f}% | F1: {f1:.1f}%")

    os.makedirs("models", exist_ok=True)
    save_path = 'models/halting_snn_weights.pt'
    torch.save(model.state_dict(), save_path)
    log_status(f"SUCCESS: Elastic Halting weights extracted to {save_path}")