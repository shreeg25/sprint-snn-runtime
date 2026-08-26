import numpy as np
import matplotlib.pyplot as plt

# SNN Hyperparameters
T_normal = 100
T_sprint = 30
tau = 0.8  # Decay factor
V_th_normal = 1.0
V_th_sprint = 0.6  # Dynamic thresholding during power collapse

# Simulate random incoming spikes from a sensor (e.g., vibration data)
np.random.seed(42)
input_spikes = np.random.choice([0, 1], size=T_normal, p=[0.7, 0.3])
weights = np.random.uniform(0.1, 0.4, size=T_normal)

def simulate_lif(T, V_th, input_spikes, weights):
    mem = np.zeros(T)
    out_spikes = np.zeros(T)
    
    for t in range(1, T):
        # LIF Equation: Decay + Input
        mem[t] = mem[t-1] * tau + (input_spikes[t] * weights[t])
        
        # Check for spike
        if mem[t] >= V_th:
            out_spikes[t] = 1
            mem[t] = 0  # Reset after spike
            
    return mem, out_spikes

# 1. Simulate Normal Operation
mem_normal, spikes_normal = simulate_lif(T_normal, V_th_normal, input_spikes, weights)

# 2. Simulate Sprint Mode (Pruned Window + Dynamic Threshold)
mem_sprint, spikes_sprint = simulate_lif(T_sprint, V_th_sprint, input_spikes[:T_sprint], weights[:T_sprint])

# --- Plotting the Results ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Normal Mode Plot
ax1.plot(mem_normal, label='Membrane Potential $V(t)$', color='blue')
ax1.axhline(V_th_normal, color='red', linestyle='--', label='Threshold ($V_{th} = 1.0$)')
spike_times_normal = np.where(spikes_normal == 1)[0]
ax1.vlines(spike_times_normal, ymin=0, ymax=V_th_normal, colors='black', label='Output Spikes')
ax1.axvline(T_sprint, color='gray', linestyle=':', label='Power Collapse (T=30)')
ax1.set_title("Standard Rate Coding (Fails if power cuts at T=30)")
ax1.set_ylabel("Voltage")
ax1.legend(loc="upper left")

# Sprint Mode Plot
ax2.plot(mem_sprint, label='Membrane Potential $V(t)$', color='purple')
ax2.axhline(V_th_sprint, color='red', linestyle='--', label='Dynamic Threshold ($V_{th} = 0.6$)')
spike_times_sprint = np.where(spikes_sprint == 1)[0]
ax2.vlines(spike_times_sprint, ymin=0, ymax=V_th_sprint, colors='black', label='Time-to-First-Spike')
ax2.set_title("SPRINT Mode: Dynamic Thresholding & TTFS")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Voltage")
ax2.legend(loc="upper left")

plt.tight_layout()
plt.savefig('sprint_snn_results.png', dpi=300)
print("[+] Simulation complete. Results saved to sprint_snn_results.png")