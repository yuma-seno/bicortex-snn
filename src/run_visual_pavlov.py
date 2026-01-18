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

def create_stimulus_images():
    img_red = Image.new('RGB', (128, 128), color='red')
    img_blue = Image.new('RGB', (128, 128), color='blue')
    return img_red, img_blue

def normalize_features(features, target_sum=60.0):
    total = np.sum(features)
    if total > 0:
        return features * (target_sum / total)
    return features

def run_experiment():
    print("=== Visual Pavlovian Conditioning (Differential Contrast Mode) ===")
    
    encoder = VisualEncoder()
    img_red, img_blue = create_stimulus_images()
    
    # 1. 特徴量抽出
    raw_red = encoder.encode(img_red)
    raw_blue = encoder.encode(img_blue)
    
    # ★修正1: 差別化処理 (Pattern Separation)
    # 赤と青の「共通する平均的な特徴」を引くことで、色特有の差異だけを抽出
    common_pattern = (raw_red + raw_blue) / 2
    diff_red = np.maximum(0, raw_red - common_pattern)
    diff_blue = np.maximum(0, raw_blue - common_pattern)
    
    # ★修正2: コントラスト強化 ($x^2$)
    # 特徴を鋭く際立たせ、脳内での「赤チーム」と「青チーム」の住み分けを強制
    feat_red = normalize_features(diff_red**2, target_sum=60.0)
    feat_blue = normalize_features(diff_blue**2, target_sum=60.0)
    
    print(f"Red unique features: {np.count_nonzero(feat_red)}/576")
    print(f"Blue unique features: {np.count_nonzero(feat_blue)}/576")
    
    vis_dim = 576
    
    # 2. SNNエンジンの初期化 (容量を200へ拡張)
    engine = BiCortexEngine(
        n_input=vis_dim + 1, n_hidden=10, n_motor=2, n_mem=200, 
        dt=1.0, learning_rate=0.005, gate_ratio=0.05,
        global_decay=0.01, w_scale_fixed=0.0, w_scale_rec=0.0,
        adaptation_beta=1.0, adaptation_tau=20.0, seed=42
    )
    
    # キレ重視設定 (v2.3)
    engine.tau_fast = 5.0
    engine.decay_fast = np.exp(-engine.dt / engine.tau_fast)
    
    print("... Wiring Anatomy ...")
    idx_vis, idx_taste, idx_salivation = np.arange(0, vis_dim), np.array([vis_dim]), engine.idx_motor[0]
    idx_mem_food, idx_mem_res = engine.idx_mem[-20:], engine.idx_mem[:-20]

    # 1. Vision -> Memory (Sparse & Distributed)
    rng = np.random.default_rng(42)
    w_vis = rng.random((len(idx_mem_res), len(idx_vis))) * 2.0 
    mask_vis = rng.random((len(idx_mem_res), len(idx_vis))) < 0.03 # 3%接続に絞る
    engine.W[np.ix_(idx_mem_res, idx_vis)] = w_vis * mask_vis

    # 2. Taste/Readout/Reflex (Fixed)
    engine.W[np.ix_(idx_mem_food, idx_taste)] = 2.0
    engine.W[np.ix_([idx_salivation], idx_mem_food)] = 1.5
    engine.W[idx_salivation, idx_taste] = 2.0
    
    # 5. Association [PLASTIC]
    engine.W[np.ix_(idx_mem_food, idx_mem_res)] = 0.0
    engine.mask_plastic[np.ix_(idx_mem_food, idx_mem_res)] = True

    print("\n--- Trial Logs ---")
    print(f"{'Trial':<6} | {'Stimulus':<8} | {'Food':<6} | {'Salivation':<10} | {'Status'}")
    print("-" * 60)

    traces = {}
    for trial in range(TRIALS):
        is_red_trial, stimulus_name = (trial % 2 == 0), "RED" if (trial % 2 == 0) else "BLUE"
        has_food = is_red_trial
        
        # Reset Brain
        engine.v[:], engine.x_fast[:], engine.e_slow[:], engine.refractory_count[:], engine.v_th_adaptive[:] = 0.0, 0.0, 0.0, 0, 0.0
        spike_record, spike_count = [], 0
        
        for t in range(STEPS_PER_TRIAL):
            input_current = np.zeros(engine.n_total)
            if 50 <= t < 150: input_current[idx_vis] = feat_red if is_red_trial else feat_blue
            if has_food and (140 <= t < 150): input_current[idx_taste] = 2.0
            spikes = engine.step(input_current, learning=True)
            spike_record.append(spikes)
            if spikes[idx_salivation] > 0: spike_count += 1
        
        status = "."
        if spike_count > 0:
            if not has_food: status = "False Alarm!"
            elif spike_count > 12: status = "Predict!"
        print(f"{trial+1:<6} | {stimulus_name:<8} | {str(has_food):<6} | {spike_count:<10} | {status}")

        motor_spikes = np.array(spike_record)[:, idx_salivation]
        if trial == 0: traces['early_red'] = motor_spikes
        if trial == 1: traces['early_blue'] = motor_spikes
        if trial == TRIALS - 2: traces['late_red'] = motor_spikes
        if trial == TRIALS - 1: traces['late_blue'] = motor_spikes

    visualize_detailed_results(traces)

def visualize_detailed_results(traces):
    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    def plot_trial(ax, data, title, stimulus_color, has_food):
        ax.plot(data, color='black', linewidth=1.5, label='Salivation Spike')
        ax.axvspan(50, 150, color=stimulus_color, alpha=0.2, label=f'{stimulus_color.capitalize()} Image')
        if has_food: ax.axvspan(140, 150, color='green', alpha=0.5, label='Food')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel("Spikes")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, linestyle='--', alpha=0.6)
    
    plot_trial(axes[0], traces['early_red'], "Trial 1: Early RED", 'red', True)
    plot_trial(axes[1], traces['early_blue'], "Trial 2: Early BLUE", 'blue', False)
    plot_trial(axes[2], traces['late_red'], "Trial 29: Late RED (Prediction)", 'red', True)
    plot_trial(axes[3], traces['late_blue'], "Trial 30: Late BLUE (Silence)", 'blue', False)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "visual_pavlov_detailed.png"))
    print(f"\nDetailed Graph saved to: {os.path.join(SAVE_DIR, 'visual_pavlov_detailed.png')}")

if __name__ == "__main__":
    run_experiment()