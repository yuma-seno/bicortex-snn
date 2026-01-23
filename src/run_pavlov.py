import numpy as np
import matplotlib.pyplot as plt
import os
from core import BiCortexEngine

# --- 設定 ---
STEPS_PER_TRIAL = 300
TRIALS = 20
SAVE_DIR = "reports/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# v13エンジンの高閾値(5.0)に対応するための強力な入力ゲイン
INPUT_GAIN = 10.0 

def run_experiment():
    print("=== Pavlovian Conditioning (Bio-Intrinsic Mode) ===")
    
    # 1. エンジンの初期化 (v13仕様)
    # n_hiddenはModulator(5つ)を含むため、最低5以上必要
    engine = BiCortexEngine(
        n_input=2,      # 0:Bell, 1:Food
        n_hidden=10,    # 0-4:Modulator, 5-9:General
        n_motor=1,      # 0:Salivation
        n_mem=50,       # Memory Cortex
        dt=1.0, 
        learning_rate=0.05,
        gate_ratio=0.05,
        global_decay=0.005,     # 緩やかな忘却
        w_scale_fixed=0.0,
        w_scale_rec=0.0,        # Traceバッファモード (リカレントなし)
        seed=42
    )
    
    # --- Anatomy & Interface Wiring ---
    print("... Wiring Anatomy ...")
    
    # ID取得
    idx_bell = engine.idx_input[0]
    idx_food = engine.idx_input[1]
    idx_salivation = engine.idx_motor[0]
    idx_modulator = engine.idx_modulator # ドーパミン作動性ニューロン群
    
    # 記憶野の割り当て (Interface Projection先の定義)
    idx_mem_bell = engine.idx_mem[:10]  # ベルの記憶痕跡用
    
    # ----------------------------------------------------
    # 1. Thinking Cortex Internal (Instincts) - 固定
    # ----------------------------------------------------
    # [A] 報酬系: Food入力 -> Modulator発火 (ドーパミン放出)
    # これにより、エサがある時だけ脳全体が学習モードになる
    engine.W[np.ix_(idx_modulator, [idx_food])] = 5.0
    
    # [B] 反射系: Food入力 -> Salivation (ヨダレ)
    engine.W[idx_salivation, idx_food] = 5.0

    # ----------------------------------------------------
    # 2. Interface Projection (Thinking -> Memory) - 固定
    # ----------------------------------------------------
    # Bell入力 -> 記憶野の特定領域へ投影
    engine.W[np.ix_(idx_mem_bell, [idx_bell])] = 5.0

    # ----------------------------------------------------
    # 3. Memory Readout (Memory -> Thinking) - 可塑
    # ----------------------------------------------------
    # 記憶野(Bell痕跡) -> 運動野(ヨダレ)
    # ここが学習対象。初期値はゼロ (Tabula Rasa)
    engine.W[np.ix_([idx_salivation], idx_mem_bell)] = 0.0
    engine.mask_plastic[np.ix_([idx_salivation], idx_mem_bell)] = True
    
    # Check
    w_check = np.mean(engine.W[np.ix_([idx_salivation], idx_mem_bell)])
    print(f"Initial Association Weight: {w_check:.6f}")
    
    print("\n--- Trial Logs ---")
    print(f"{'Trial':<6} | {'Bell':<8} | {'Food':<8} | {'Assoc W':<12} | {'Status'}")
    print("-" * 60)

    history_motor = []
    history_weights = []

    for trial in range(TRIALS):
        # Reset States (短期記憶のみリセット)
        engine.reset_state()
        
        spike_record = []
        bell_spikes = 0
        food_spikes = 0
        
        for t in range(STEPS_PER_TRIAL):
            input_current = np.zeros(engine.n_total)
            
            # Scenario
            # Bell: 予兆 (t=50-60)
            if 50 <= t < 60:
                input_current[idx_bell] = INPUT_GAIN
            
            # Food: 報酬 (t=150-160)
            if 150 <= t < 160:
                input_current[idx_food] = INPUT_GAIN

            # Step (自律学習)
            spikes = engine.step(input_current)
            spike_record.append(spikes)
            
            # 運動野の発火カウント
            if spikes[idx_salivation] > 0:
                if 50 <= t < 100: bell_spikes += 1  # 予期反応
                if 150 <= t < 200: food_spikes += 1 # 反射反応

        history_motor.append(np.array(spike_record)[:, idx_salivation])
        
        # 重みの平均値を記録
        current_w = np.mean(engine.W[np.ix_([idx_salivation], idx_mem_bell)])
        history_weights.append(current_w)
        
        status = "."
        if bell_spikes > 0: status = "Predict!"
        
        print(f"{trial+1:<6} | {bell_spikes:<8} | {food_spikes:<8} | {current_w:.4f}       | {status}")

    visualize_results(history_motor, history_weights)

def visualize_results(motor_activity, weight_history):
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Initial Trial
    axes[0].plot(motor_activity[0], label="Trial 1", color='blue')
    axes[0].axvspan(50, 60, color='gray', alpha=0.3, label="Bell")
    axes[0].axvspan(150, 160, color='green', alpha=0.3, label="Food")
    axes[0].set_title("Trial 1: Motor Response (Reflex Only)")
    axes[0].set_ylabel("Spikes")
    axes[0].legend()
    axes[0].set_ylim(-0.1, 1.1)

    # Plot 2: Final Trial
    axes[1].plot(motor_activity[-1], label="Trial 20", color='red')
    axes[1].axvspan(50, 60, color='gray', alpha=0.3, label="Bell")
    axes[1].axvspan(150, 160, color='green', alpha=0.3, label="Food")
    axes[1].set_title("Trial 20: Motor Response (Learned Association)")
    axes[1].set_ylabel("Spikes")
    axes[1].legend()
    axes[1].set_ylim(-0.1, 1.1)

    # Plot 3: Weight Evolution
    axes[2].plot(weight_history, marker='o', color='purple')
    axes[2].set_title("Association Strength (Memory -> Motor)")
    axes[2].set_xlabel("Trial")
    axes[2].set_ylabel("Mean Weight")
    axes[2].grid(True)
    axes[2].set_ylim(bottom=0.0)

    plt.tight_layout()
    output_path = os.path.join(SAVE_DIR, "pavlov_result.png")
    plt.savefig(output_path)
    print(f"\nGraph saved to: {output_path}")

if __name__ == "__main__":
    run_experiment()