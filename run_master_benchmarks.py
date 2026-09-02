import os
import sys
import torch
import torch.nn as nn
import torch.ao.quantization
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import dataset and architecture directly from your SNN source
from train_snn import MITBIHSpikeDataset, WearableSNN

def log_status(msg):
    sys.stdout.write(f"[BENCHMARK] {msg}\n")
    sys.stdout.flush()

# --- 1. Continuous Baseline Architectures ---
class BaselineMLP(nn.Module):
    def __init__(self):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(100, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.squeeze(-1)
        return self.fc2(self.relu(self.fc1(x)))

class QuantizableCNN(nn.Module):
    def __init__(self):
        super(QuantizableCNN, self).__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(16 * 48, 2)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.flatten(x)
        x = self.fc(x)
        return self.dequant(x)

# --- 2. Visualization Engine ---
def generate_master_plot(df):
    log_status("Rendering IEEE-compliant architecture comparison graphic...")
    plt.rcParams.update({'font.weight': 'bold', 'axes.labelweight': 'bold', 'font.size': 11})
    
    metrics = ['Accuracy', 'Recall', 'Precision', 'F1_Score']
    x = np.arange(len(metrics))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 7))
    
    snn = df[df['Model'] == 'Proposed SNN (Edge)'][metrics].values[0]
    mlp = df[df['Model'] == 'Float32 MLP (20 Ep)'][metrics].values[0]
    cnn_f32 = df[df['Model'] == 'Float32 CNN (20 Ep)'][metrics].values[0]
    cnn_int8 = df[df['Model'] == 'INT8 CNN (Edge)'][metrics].values[0]
    rf = df[df['Model'] == 'Random Forest (Edge)'][metrics].values[0]

    # High-contrast, publication-grade palette
    ax.bar(x - 2.0*width, snn, width, label='Proposed 1-bit SNN', color='#d62728', edgecolor='black')
    ax.bar(x - 1.0*width, mlp, width, label='Baseline Float32 MLP', color='#1f77b4', edgecolor='black')
    ax.bar(x, cnn_f32, width, label='Baseline Float32 CNN', color='#9467bd', edgecolor='black')
    ax.bar(x + 1.0*width, cnn_int8, width, label='Baseline INT8 CNN', color='#7f7f7f', edgecolor='black')
    ax.bar(x + 2.0*width, rf, width, label='Baseline Random Forest', color='#2ca02c', edgecolor='black')

    ax.set_ylabel('PERCENTAGE (%)', fontweight='bold')
    ax.set_title('ARCHITECTURAL BENCHMARK: 20-EPOCH SYNCHRONIZED COMPARISON', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics], fontweight='bold')
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right', framealpha=1.0, edgecolor='black', ncol=2)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plot_dir = os.path.join(os.getcwd(), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    out_path = os.path.join(plot_dir, 'master_architecture_comparison.pdf')
    plt.savefig(out_path, format='pdf', dpi=300)
    log_status(f"Saved figure to {out_path}")

# --- 3. Master Execution Engine ---
if __name__ == "__main__":
    device = torch.device("cpu")
    torch.backends.quantized.engine = 'qnnpack'
    EPOCHS = 20
    
    log_status("Initializing Global Dataset across 48 patients...")
    dataset = MITBIHSpikeDataset(data_dir='data/raw', window_size=100, threshold=0.15)
    class_counts = np.bincount(dataset.labels)
    sample_weights = [1.0 / class_counts[label] for label in dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataset), replacement=True)
    dataloader = DataLoader(dataset, batch_size=64, sampler=sampler)

    results = []

    # -------------------------------------------------------------
    # 1. EVALUATE PROPOSED SNN (20 Epochs Trained)
    # -------------------------------------------------------------
    log_status("Evaluating Proposed Wearable SNN from 20-Epoch weights...")
    snn_model = WearableSNN(num_hidden=256).to(device)
    weight_path = 'models/distilled_snn_weights.pt'
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"CRITICAL: {weight_path} missing. Run train_snn.py first.")
        
    snn_model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    snn_model.eval()
    
    t_all_snn, p_all_snn = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.transpose(0, 1).to(device)
            _, mem_rec = snn_model(data)
            _, preds = mem_rec.sum(dim=0).max(1)
            t_all_snn.extend(targets.cpu().numpy())
            p_all_snn.extend(preds.cpu().numpy())
            
    results.append({
        'Model': 'Proposed SNN (Edge)', 
        'Accuracy': accuracy_score(t_all_snn, p_all_snn)*100, 
        'Recall': recall_score(t_all_snn, p_all_snn, zero_division=0)*100, 
        'Precision': precision_score(t_all_snn, p_all_snn, zero_division=0)*100, 
        'F1_Score': f1_score(t_all_snn, p_all_snn, zero_division=0)*100
    })

    # -------------------------------------------------------------
    # 2. TRAIN & EVALUATE BASELINE MLP (20 Epochs)
    # -------------------------------------------------------------
    log_status(f"Training Baseline MLP for {EPOCHS} epochs (Full Convergence)...")
    mlp = BaselineMLP().to(device)
    optimizer_mlp = optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        mlp.train()
        for data, targets in dataloader:
            optimizer_mlp.zero_grad()
            loss = criterion(mlp(data.to(device)), targets.to(device))
            loss.backward()
            optimizer_mlp.step()
            
    mlp.eval()
    t_all, p_all = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            _, preds = torch.max(mlp(data.to(device)), 1)
            t_all.extend(targets.numpy())
            p_all.extend(preds.numpy())
    
    results.append({
        'Model': 'Float32 MLP (20 Ep)', 
        'Accuracy': accuracy_score(t_all, p_all)*100, 
        'Recall': recall_score(t_all, p_all, zero_division=0)*100, 
        'Precision': precision_score(t_all, p_all, zero_division=0)*100, 
        'F1_Score': f1_score(t_all, p_all, zero_division=0)*100
    })

    # -------------------------------------------------------------
    # 3. TRAIN BASELINE CNN (20 Epochs) & EVALUATE Float32 vs INT8
    # -------------------------------------------------------------
    log_status(f"Training Baseline 1D-CNN for {EPOCHS} epochs in Float32...")
    cnn = QuantizableCNN().to(device)
    optimizer_cnn = optim.Adam(cnn.parameters(), lr=1e-3)
    
    for epoch in range(EPOCHS):
        cnn.train()
        for data, targets in dataloader:
            optimizer_cnn.zero_grad()
            loss = criterion(cnn(data.to(device)), targets.to(device))
            loss.backward()
            optimizer_cnn.step()
            
    # Float32 Evaluation
    cnn.eval()
    t_all_f32, p_all_f32 = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            _, preds = torch.max(cnn(data.to(device)), 1)
            t_all_f32.extend(targets.numpy())
            p_all_f32.extend(preds.numpy())
            
    results.append({
        'Model': 'Float32 CNN (20 Ep)', 
        'Accuracy': accuracy_score(t_all_f32, p_all_f32)*100, 
        'Recall': recall_score(t_all_f32, p_all_f32, zero_division=0)*100, 
        'Precision': precision_score(t_all_f32, p_all_f32, zero_division=0)*100, 
        'F1_Score': f1_score(t_all_f32, p_all_f32, zero_division=0)*100
    })

    # Post-Training Quantization (PTQ) to INT8
    log_status("Calibrating & Quantizing trained CNN to INT8...")
    cnn.qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
    torch.ao.quantization.prepare(cnn, inplace=True)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            cnn(data)
            if i >= 15: break
            
    torch.ao.quantization.convert(cnn, inplace=True)
    
    t_all_int8, p_all_int8 = [], []
    with torch.no_grad():
        for data, targets in dataloader:
            _, preds = torch.max(cnn(data), 1)
            t_all_int8.extend(targets.numpy())
            p_all_int8.extend(preds.numpy())
            
    results.append({
        'Model': 'INT8 CNN (Edge)', 
        'Accuracy': accuracy_score(t_all_int8, p_all_int8)*100, 
        'Recall': recall_score(t_all_int8, p_all_int8, zero_division=0)*100, 
        'Precision': precision_score(t_all_int8, p_all_int8, zero_division=0)*100, 
        'F1_Score': f1_score(t_all_int8, p_all_int8, zero_division=0)*100
    })

    # -------------------------------------------------------------
    # 4. TRAIN & EVALUATE CLASSICAL RANDOM FOREST
    # -------------------------------------------------------------
    log_status("Training Classical Random Forest Baseline (50 Trees)...")
    X_all, y_all = [], []
    for data, targets in dataloader:
        X_all.append(data.squeeze().numpy())
        y_all.append(targets.numpy())
    X_all, y_all = np.concatenate(X_all), np.concatenate(y_all)
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X_all, y_all)
    rf_preds = rf.predict(X_all)
    
    results.append({
        'Model': 'Random Forest (Edge)', 
        'Accuracy': accuracy_score(y_all, rf_preds)*100, 
        'Recall': recall_score(y_all, rf_preds, zero_division=0)*100, 
        'Precision': precision_score(y_all, rf_preds, zero_division=0)*100, 
        'F1_Score': f1_score(y_all, rf_preds, zero_division=0)*100
    })

    # -------------------------------------------------------------
    # 5. LOGGING & GRAPHIC EXPORT
    # -------------------------------------------------------------
    df = pd.DataFrame(results)
    
    csv_path = os.path.join(os.getcwd(), 'comprehensive_baselines.csv')
    df.to_csv(csv_path, index=False)
    
    txt_path = os.path.join(os.getcwd(), 'comprehensive_baselines.txt')
    with open(txt_path, 'w') as f:
        f.write("SYNCHRONIZED 20-EPOCH ARCHITECTURAL BENCHMARK\n")
        f.write("=============================================\n")
        for index, row in df.iterrows():
            f.write(f"Model: {row['Model']}\n")
            f.write(f"  -> Accuracy:  {row['Accuracy']:.2f}%\n")
            f.write(f"  -> Recall:    {row['Recall']:.2f}%\n")
            f.write(f"  -> Precision: {row['Precision']:.2f}%\n")
            f.write(f"  -> F1-Score:  {row['F1_Score']:.2f}%\n")
            f.write("---------------------------------------------\n")
            
    log_status(f"Exported metrics to {csv_path} and {txt_path}")
    generate_master_plot(df)