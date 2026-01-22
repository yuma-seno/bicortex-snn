import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from core import BiCortexEngine
from core.visual import VisualEncoder

# --- 設定 ---
STEPS_PER_TRIAL = 300
TRIALS = 30 
SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)
INPUT_GAIN = 50.0 

class Colors:
    RESET = "\033[0m"
    LOW = "\033[90m"
    MED = "\033[33m"
    HIGH = "\033[31m"
    FOOD = "\033[92m"

class InstinctSystem:
    def __init__(self, idx_reward_neurons):
        self.idx_r = idx_reward_neurons
        self.reward_scale = 10.0

    def get_dopamine(self, spikes):
        r_activity = np.sum(spikes[self.idx_r])
        return r_activity * self.reward_scale

def calibrate_interface(raw_red, raw_blue, top_k=50):
    dim = len(raw_red)
    
    def normalize(v):
        n = np.linalg.norm(v)
        return v if n == 0 else v / n

    raw_red = normalize(raw_red)
    raw_blue = normalize(raw_blue)
    
    # Contrastive Subtraction
    unique_red = np.maximum(0, raw_red - raw_blue)
    unique_blue = np.maximum(0, raw_blue - raw_red)
    
    # Noise Floor Cut
    unique_red[unique_red < 0.05] = 0.0
    unique_blue[unique_blue < 0.05] = 0.0
    
    vec_red = np.zeros(dim)
    idx_red_top = np.argsort(unique_red)[-top_k:]
    vec_red[idx_red_top] = unique_red[idx_red_top]
    
    vec_blue = np.zeros(dim)
    idx_blue_top = np.argsort(unique_blue)[-top_k:]
    vec_blue[idx_blue_top] = unique_blue[idx_blue_top]
    
    return normalize(vec_red), normalize(vec_blue)

def create_stimulus_images():
    return Image.new('RGB', (128, 128), color='red'), Image.new('RGB', (128, 128), color='blue')

def get_activity_heatmap_str(spike_history, bin_size=20):
    steps = len(spike_history)
    heatmap_str = ""
    for t in range(0, steps, bin_size):
        count = np.sum(spike_history[t : t+bin_size])
        char = "."
        color = Colors.LOW
        if count > 0:
            char = str(int(count)) if count < 10 else "*"
            if count <= 2: color = Colors.LOW
            elif count <= 5: color = Colors.MED
            else: color = Colors.HIGH
        
        if 140 <= t < 150: 
             heatmap_str += f"{Colors.FOOD}|{color}{char}{Colors.FOOD}|{Colors.RESET}"
        else:
             heatmap_str += f"{color}{char}{Colors.RESET}"
    return heatmap_str

def run_experiment():
    print("=== Visual Pavlovian Conditioning (No Recurrence) ===")
    
    encoder = VisualEncoder()
    vis_dim = 576
    img_red, img_blue = create_stimulus_images()

    print("Phase 1: Thinking Cortex Pre-processing...")
    raw_red = encoder.encode(img_red)
    raw_blue = encoder.encode(img_blue)
    
    vec_red, vec_blue = calibrate_interface(raw_red, raw_blue, top_k=50)
    
    crosstalk = np.dot(vec_red, raw_blue / np.linalg.norm(raw_blue))
    print(f"Interface Crosstalk Level: {crosstalk:.6f}")
    
    idx_taste_local = vis_dim 
    n_hidden = 10 
    
    engine = BiCortexEngine(
        n_input=vis_dim + 1, 
        n_hidden=n_hidden, 
        n_motor=2, 
        n_mem=500,
        dt=1.0, 
        learning_rate=0.05, 
        gate_ratio=0.02,
        global_decay=0.001, 
        # ★修正: リカレント結合をゼロにする。
        # これにより、Memory Cortex内の発火が他のニューロンへ伝播(汚染)するのを防ぐ。
        w_scale_rec=0.0, 
        seed=42
    )
    
    engine.v_base = 5.0 
    engine.tau_fast = 5.0
    engine.decay_fast = np.exp(-engine.dt / engine.tau_fast)

    print("Phase 2: Wiring & Interface Setup...")
    idx_vis = np.arange(0, vis_dim)
    idx_taste = np.array([vis_dim])
    idx_salivation = engine.idx_motor[0]
    
    idx_mem_assoc = engine.idx_mem  
    idx_reward_neurons = engine.idx_hidden[0:5] 
    
    # Interface Wiring
    n_conns = min(len(idx_mem_assoc), vis_dim)
    for i in range(n_conns):
        target = idx_mem_assoc[i]
        src = idx_vis[i]
        engine.W[target, src] = 5.0
        
    # Instinct Wiring
    engine.W[np.ix_(idx_reward_neurons, idx_taste)] = 5.0
    engine.W[idx_salivation, idx_taste] = 5.0

    # Memory Plasticity
    engine.W[np.ix_([idx_salivation], idx_mem_assoc)] = 0.0
    engine.mask_plastic[np.ix_([idx_salivation], idx_mem_assoc)] = True
    
    instinct = InstinctSystem(idx_reward_neurons)

    print("\nPhase 3: Online Learning Loop")
    print("-" * 80)
    print(f"{'Trial':<6} | {'Stim':<5} | {'Motor Activity Heatmap':<40} | {'Status'}")
    print("-" * 80)

    traces = {}

    for trial in range(TRIALS):
        is_red_trial = (trial % 2 == 0)
        stimulus_name = "RED" if is_red_trial else "BLUE"
        has_food = is_red_trial 
        
        engine.reset_state()
        spike_record = []
        
        for t in range(STEPS_PER_TRIAL):
            input_current = np.zeros(engine.n_total)
            
            if 50 <= t < 150:
                if is_red_trial:
                    input_current[idx_vis] = vec_red * INPUT_GAIN 
                else:
                    input_current[idx_vis] = vec_blue * INPUT_GAIN 
            
            if has_food and (140 <= t < 150):
                input_current[idx_taste] = 4.0 
            
            spikes = engine.step(input_current)
            dopamine = instinct.get_dopamine(spikes)
            engine.update_weights(spikes, dopamine)
            
            spike_record.append(spikes)
        
        motor_spikes = np.array(spike_record)[:, idx_salivation]
        heatmap = get_activity_heatmap_str(motor_spikes)
        prefood_spikes = np.sum(motor_spikes[50:140])
        
        status = ""
        if prefood_spikes > 5:
            status = "Predict!" if is_red_trial else "False Alarm"
        
        print(f"{trial+1:<6} | {stimulus_name:<5} | {heatmap} | {status}")

        if trial == 0: traces['early_red'] = motor_spikes
        if trial == 1: traces['early_blue'] = motor_spikes
        if trial == TRIALS - 2: traces['late_red'] = motor_spikes
        if trial == TRIALS - 1: traces['late_blue'] = motor_spikes

    visualize_detailed_results(traces)

def visualize_detailed_results(traces):
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    def plot_trial(ax, data, title, color, food):
        ax.plot(data, color='black', linewidth=1.5)
        ax.axvspan(50, 150, color=color, alpha=0.2, label='Visual')
        if food: ax.axvspan(140, 150, color='green', alpha=0.5, label='Food')
        ax.set_title(title, fontweight='bold')
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

    if 'early_red' in traces: plot_trial(axes[0], traces['early_red'], "Trial 1: Early RED", 'red', True)
    if 'early_blue' in traces: plot_trial(axes[1], traces['early_blue'], "Trial 2: Early BLUE", 'blue', False)
    if 'late_red' in traces: plot_trial(axes[2], traces['late_red'], "Trial 29: Late RED (Prediction)", 'red', True)
    if 'late_blue' in traces: plot_trial(axes[3], traces['late_blue'], "Trial 30: Late BLUE (Silence)", 'blue', False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "visual_pavlov_detailed.png"))
    print(f"\nDetailed Graph saved to: {os.path.join(SAVE_DIR, 'visual_pavlov_detailed.png')}")

if __name__ == "__main__":
    run_experiment()