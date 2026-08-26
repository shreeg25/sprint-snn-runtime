import torch
import torch.nn as nn

class TemporalDistillationLoss(nn.Module):
    def __init__(self, temporal_penalty_gamma=0.01):
        super(TemporalDistillationLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.gamma = temporal_penalty_gamma

    def forward(self, spike_trains, targets):
        """
        spike_trains shape: [time_steps, batch_size, num_classes]
        targets shape: [batch_size]
        """
        T, B, C = spike_trains.shape
        
        # 1. Base Classification Loss (Rate Coding Baseline)
        # We sum the spikes over the entire window to see if it guessed right
        spike_count = spike_trains.sum(dim=0) 
        loss_ce = self.ce_loss(spike_count, targets)
        
        # 2. The Research Contribution: Temporal Penalty
        # Create a time vector: [1, 2, 3, ..., T]
        time_vector = torch.arange(1, T + 1, dtype=torch.float32, device=spike_trains.device)
        time_vector = time_vector.view(T, 1, 1) # Reshape for broadcasting against spikes
        
        # Multiply every spike by the exact time step it occurred. 
        # A spike at t=90 gets punished 9x harder than a spike at t=10.
        late_spike_penalty = torch.sum(spike_trains * time_vector) / B
        
        # 3. Total Loss
        total_loss = loss_ce + (self.gamma * late_spike_penalty)
        
        return total_loss, loss_ce, late_spike_penalty

import snntorch as snn
from snntorch import surrogate
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

# --- 1. SNN Architecture Definition ---
class WearableSNN(nn.Module):
    def __init__(self, num_inputs=1, num_hidden=32, num_outputs=2):
        super(WearableSNN, self).__init__()
        
        # Surrogate gradient for backprop through discrete spikes
        spike_grad = surrogate.fast_sigmoid(slope=25)
        
        # Synaptic layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.8, spike_grad=spike_grad)
        
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.8, spike_grad=spike_grad)

    def forward(self, x):
        """
        x shape: [time_steps, batch_size, features]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = []
        mem2_rec = []

        # Iterate through time steps
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# --- 2. Synthetic Data Loader (Replace with real MIT-BIH tensors later) ---
def get_synthetic_dataloader(time_steps=100, batch_size=16, samples=128):
    # Random binary spikes: [T, Batch, Features]
    X = torch.randint(0, 2, (time_steps, samples, 1), dtype=torch.float32)
    # Binary targets (0: Normal, 1: Arrhythmia)
    y = torch.randint(0, 2, (samples,), dtype=torch.long) 
    
    dataset = TensorDataset(X.transpose(0, 1), y) # Transpose to [Batch, T, Features]
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- 3. The Execution Loop ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Executing on: {device}")

    # Hyperparameters
    TIME_STEPS = 100
    EPOCHS = 5
    
    model = WearableSNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Instantiate your custom loss function
    criterion = TemporalDistillationLoss(temporal_penalty_gamma=0.05)
    
    dataloader = get_synthetic_dataloader(time_steps=TIME_STEPS)

    print("[*] Initiating Temporal Distillation Training...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        
        for batch_idx, (data, targets) in enumerate(dataloader):
            data = data.transpose(0, 1).to(device) # Reshape back to [T, Batch, Features]
            targets = targets.to(device)
            
            # Forward Pass
            spk_rec, _ = model(data)
            
            # Calculate Total Loss + Late Spike Penalty
            loss, ce, late_penalty = criterion(spk_rec, targets)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {total_loss:.4f}")

    # --- 4. Extract the Physical Silicon State ---
    torch.save(model.state_dict(), 'models/distilled_snn_weights.pt')
    print("[+] Weights heavily biased for early TTFS extracted to models/distilled_snn_weights.pt")