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

INPUT_GAIN = 1.5
REWARD_GAIN = 1.0 # ドーパミン強度

def sparsify_features(features, keep_ratio=0.1):
    if keep_ratio <= 0 or keep_ratio >= 1: return features
    flat = features.flatten()
    k = max(1, int(len(flat) * keep_ratio))
    thresh = np.partition(flat, -k)[-k]
    return np.where(features >= thresh, features, 0.0)

def create_stimulus_images():
    return Image.new('RGB', (128, 128), color='red'), Image.new('RGB', (128, 128), color='blue')

def normalize_features(features, target_sum=150.0):
    total = np.sum(features)
    if total > 0: return features * (target_sum / total)
    return features

def run_experiment():
    print("=== Visual Pavlovian Conditioning (Integrated 3-Factor Learning) ===")
    
    encoder = VisualEncoder()
    img_red, img_blue = create_stimulus_images()
    vis_dim = 576

    # 1. 特徴量抽出 (ユーザー様のロジックを維持)
    raw_red = encoder.encode(img_red)
    raw_blue = encoder.encode(img_blue)
    common_pattern = (raw_red + raw_blue) / 2
    diff_red = np.maximum(0, raw_red - common_pattern)
    diff_blue = np.maximum(0, raw_blue - common_pattern)
    
    red_feat = diff_red**2
    blue_feat = diff_blue**2
    feat_red = np.maximum(0, red_feat - blue_feat)
    feat_blue = np.maximum(0, blue_feat - red_feat)
    feat_red = sparsify_features(feat_red, keep_ratio=0.1)
    feat_blue = sparsify_features(feat_blue, keep_ratio=0.1)

    feat_red = normalize_features(feat_red, target_sum=150.0)
    feat_blue = normalize_features(feat_blue, target_sum=150.0)
    
    # 2. Engine
    engine = BiCortexEngine(
        n_input=vis_dim + 1, n_hidden=10, n_motor=2, n_mem=1000,
        dt=1.0, learning_rate=0.01, gate_ratio=0.05, # 学習率は内部で使うので適正値に
        global_decay=0.005, w_scale_fixed=0.0, w_scale_rec=0.0,
        adaptation_beta=0.3, adaptation_tau=30.0, seed=42
    )
    engine.v_base = 1.0
    engine.tau_fast = 5.0
    engine.decay_fast = np.exp(-engine.dt / engine.tau_fast)
    
    print("... Wiring Anatomy ...")
    idx_vis = np.arange(0, vis_dim)
    idx_taste = np.array([vis_dim])
    idx_salivation = engine.idx_motor[0]
    idx_mem_food, idx_mem_res = engine.idx_mem[:100], engine.idx_mem[100:]

    # Wiring
    rng = np.random.default_rng(42)
    w_vis = rng.random((len(idx_mem_res), len(idx_vis))) * 6.0
    mask_vis = rng.random((len(idx_mem_res), len(idx_vis))) < 0.002
    engine.W[np.ix_(idx_mem_res, idx_vis)] = w_vis * mask_vis

    engine.W[np.ix_(idx_mem_food, idx_taste)] = 2.0
    engine.W[np.ix_([idx_salivation], idx_mem_food)] = 1.0
    engine.W[idx_salivation, idx_taste] = 1.2
    
    # Plasticity Targets
    engine.W[np.ix_(idx_mem_food, idx_mem_res)] = 0.0
    engine.mask_plastic[np.ix_(idx_mem_food, idx_mem_res)] = True
    # 出力層への結合も学習させる
    engine.mask_plastic[np.ix_([idx_salivation], idx_mem_food)] = True

    print("\n--- Trial Logs ---")
    print(f"{'Trial':<6} | {'Stimulus':<8} | {'Food':<6} | {'PreFood':<8} | {'Total':<6} | {'Status'}")
    print("-" * 60)

    traces = {}
    for trial in range(TRIALS):
        is_red_trial, stimulus_name = (trial % 2 == 0), "RED" if (trial % 2 == 0) else "BLUE"
        has_food = is_red_trial
        
        engine.v[:] = 0; engine.refractory_count[:] = 0
        spike_record, spike_count = [], 0
        prefood_count = 0
        
        for t in range(STEPS_PER_TRIAL):
            input_current = np.zeros(engine.n_total)
            if 50 <= t < 150:
                input_current[idx_vis] = INPUT_GAIN * (feat_red if is_red_trial else feat_blue)
            if has_food and (140 <= t < 150): input_current[idx_taste] = 2.0
            
            # --- 報酬信号の生成 ---
            reward = 0.0
            
            # A. 正の報酬 (Food Reward): エサが出たら強化
            if has_food and (140 <= t < 150):
                reward = REWARD_GAIN # +1.0
            
            # B. 負の報酬 (False Alarm Punishment): エサがないのにヨダレが出たら罰する
            is_false_alarm_context = (not has_food) and (50 <= t < 150)
            if is_false_alarm_context:
                reward = -0.1 # 軽微な罰 (抑制)
            
            # Step実行 (学習含む)
            spikes = engine.step(input_current, reward=reward, learning=True)
            
            # 罰則の追加ロジック:
            # もし「今まさにFalse Alarm発火した」なら、強い罰を与える
            if is_false_alarm_context and spikes[idx_salivation] > 0:
                # 強い罰を与えて再更新（同じステップで二度更新になるが許容）
                engine._update_weights(spikes, reward=-1.0)

            spike_record.append(spikes)
            if spikes[idx_salivation] > 0:
                spike_count += 1
                if 90 <= t < 140: prefood_count += 1
        
        status = "."
        if prefood_count > 0:
            if not has_food: status = "False Alarm!"
            elif prefood_count >= 3: status = "Predict!"
        print(f"{trial+1:<6} | {stimulus_name:<8} | {str(has_food):<6} | {prefood_count:<8} | {spike_count:<6} | {status}")

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