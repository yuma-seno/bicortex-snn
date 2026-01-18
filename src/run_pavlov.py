import numpy as np
import matplotlib.pyplot as plt
import os
from core import BiCortexEngine

# --- 設定 ---
STEPS_PER_TRIAL = 300
TRIALS = 20
SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

def run_experiment():
    print("=== Pavlovian Conditioning (Perfect Timing Mode) ===")
    
    # 1. エンジンの初期化
    engine = BiCortexEngine(
        n_input=2, n_hidden=10, n_motor=2, n_mem=50, 
        dt=1.0, 
        learning_rate=0.005,
        gate_ratio=0.05,
        # ★修正1: 忘却をOFFにする (前回はこれが学習を阻害していた)
        global_decay=0.0,
        w_scale_fixed=0.0,
        w_scale_rec=0.0,        # 完全切断
        # ★修正2: 適度なブレーキ (1.0は強すぎたので0.5へ)
        adaptation_beta=0.5,
        adaptation_tau=20.0,    # 回復は速めに
        seed=42
    )
    
    # ★修正3: シナプス電流のキレを良くする
    # デフォルト20msだと残響が長すぎるため、5msに短縮
    engine.tau_fast = 5.0
    engine.decay_fast = np.exp(-engine.dt / engine.tau_fast)
    
    # 膜電位の減衰は標準(20ms)に戻す (学習に必要な「予期の重なり」を作るため)
    engine.tau_m = 20.0
    engine.alpha = np.exp(-engine.dt / engine.tau_m)
    
    # --- Anatomy ---
    print("... Wiring Anatomy ...")
    idx_bell = engine.idx_input[0]
    idx_food = engine.idx_input[1]
    idx_salivation = engine.idx_motor[0]
    
    idx_mem_bell = engine.idx_mem[:10]
    idx_mem_food = engine.idx_mem[40:50]

    # Fixed Wiring
    engine.W[np.ix_(idx_mem_bell, [idx_bell])] = 1.5
    engine.W[np.ix_(idx_mem_food, [idx_food])] = 1.5
    engine.W[np.ix_([idx_salivation], idx_mem_food)] = 1.5
    engine.W[idx_salivation, idx_food] = 1.5 # Reflex
    
    if len(engine.idx_mem_inh) > 0:
        engine.W[:, engine.idx_mem_inh] *= 5.0

    # Plastic Wiring (Bell -> Food)
    engine.W[np.ix_(idx_mem_food, idx_mem_bell)] = 0.0
    engine.mask_plastic[np.ix_(idx_mem_food, idx_mem_bell)] = True
    
    # Check
    w_check = np.mean(engine.W[np.ix_(idx_mem_food, idx_mem_bell)])
    print(f"Initial Association Weight: {w_check:.6f}")
    
    print("\n--- Trial Logs ---")
    print(f"{'Trial':<6} | {'Bell':<8} | {'Food':<8} | {'Assoc W':<12} | {'Status'}")
    print("-" * 60)

    history_motor = []
    history_weights = []

    for trial in range(TRIALS):
        # Reset States
        engine.v[:] = 0.0
        engine.x_fast[:] = 0.0
        engine.e_slow[:] = 0.0
        engine.refractory_count[:] = 0
        engine.v_th_adaptive[:] = 0.0
        
        spike_record = []
        bell_spikes = 0
        food_spikes = 0
        
        for t in range(STEPS_PER_TRIAL):
            input_current = np.zeros(engine.n_total)
            
            # Scenario
            if 50 <= t < 60:
                input_current[idx_bell] = 1.5
            if 150 <= t < 160:
                input_current[idx_food] = 1.5

            spikes = engine.step(input_current, learning=True)
            spike_record.append(spikes)
            
            if spikes[idx_salivation] > 0:
                if 50 <= t < 100: bell_spikes += 1
                if 150 <= t < 200: food_spikes += 1

        history_motor.append(np.array(spike_record)[:, idx_salivation])
        
        current_w = np.mean(engine.W[np.ix_(idx_mem_food, idx_mem_bell)])
        history_weights.append(current_w)
        
        status = "."
        if bell_spikes > 0: status = "Predict!"
        
        # Leak Check: 刺激終了(160) + 余韻(20) = 180以降は静かであれ
        late_spikes = np.sum(np.array(spike_record)[180:, idx_salivation])
        if late_spikes > 0: status += " (Leak)"
        
        print(f"{trial+1:<6} | {bell_spikes:<8} | {food_spikes:<8} | {current_w:.4f}       | {status}")

    visualize_results(history_motor, history_weights)

def visualize_results(motor_activity, weight_history):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1
    axes[0].plot(motor_activity[0], label="Trial 1", color='blue')
    axes[0].axvspan(50, 60, color='gray', alpha=0.3, label="Bell")
    axes[0].axvspan(150, 160, color='green', alpha=0.3, label="Food")
    axes[0].set_title("Trial 1: Motor Response")
    axes[0].set_ylabel("Spikes")
    axes[0].legend()

    # Plot 2
    axes[1].plot(motor_activity[-1], label="Trial 20", color='red')
    axes[1].axvspan(50, 60, color='gray', alpha=0.3, label="Bell")
    axes[1].axvspan(150, 160, color='green', alpha=0.3, label="Food")
    axes[1].set_title("Trial 20: Motor Response")
    axes[1].set_ylabel("Spikes")
    axes[1].legend()

    # Plot 3
    axes[2].plot(weight_history, marker='o', color='purple')
    axes[2].set_title("Association Weight (Bell -> Food)")
    axes[2].set_xlabel("Trial")
    axes[2].grid(True)
    axes[2].set_ylim(bottom=0.0)

    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, "pavlov_result.png")
    plt.savefig(output_path)
    print(f"\nGraph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()