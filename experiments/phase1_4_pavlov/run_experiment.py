import sys
import os
import numpy as np
import matplotlib
# GUIのない環境でのエラー("Authorization required")を防ぐためAggバックエンドを使用
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# プロジェクトルートへのパスを通す (../../src を参照)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(os.path.join(project_root, 'src'))

from core.engine import BiCortexEngine
from utils.cli_plotter import print_cli_heatmap

def run_pavlov_experiment():
    print("=== Phase 1.4: Pavlov's Dog Experiment (Corrected) ===")
    
    # ---------------------------------------------------------
    # 1. 初期化 (Configuration)
    # ---------------------------------------------------------
    N_SENSORY = 2  # [0]: Bell, [1]: Food
    N_CONCEPT = 2  # [0]: Bell_Concept, [1]: Food_Concept
    N_MOTOR   = 1  # [0]: Salivation
    N_MEMORY  = 100 
    DT = 1.0 # ms
    
    engine = BiCortexEngine(
        n_sensory=N_SENSORY,
        n_concept=N_CONCEPT,
        n_motor=N_MOTOR,
        n_mem=N_MEMORY,
        dt=DT,
        learning_rate=0.05,   # 学習率
        gate_ratio=0.5,
        global_decay=0.001,   # 忘却/安定化用
        seed=42
    )
    engine.init_memory_reservoir(density=0.2, spectral_radius=0.95)

    # ---------------------------------------------------------
    # 2. 思考野の構築 (Hard-coded)
    # ---------------------------------------------------------
    idx_s = engine.idx_sensory
    idx_c = engine.idx_concept
    idx_m = engine.idx_motor
    idx_mem = engine.idx_mem
    
    w_strong = 10.0
    
    # [Fixed] Sensory -> Concept
    engine.W[idx_c[0], idx_s[0]] = w_strong # Bell -> BellConcept
    engine.W[idx_c[1], idx_s[1]] = w_strong # Food -> FoodConcept
    
    # [Fixed] Concept -> Motor (Reflex)
    engine.W[idx_m[0], idx_c[1]] = w_strong # FoodConcept -> Salivation
    
    # [Fixed] Concept -> Memory (Context Injection)
    rng = np.random.default_rng(42)
    engine.W[idx_mem[0]:idx_mem[-1]+1, idx_c[0]:idx_c[-1]+1] = rng.uniform(0, 1.0, (N_MEMORY, N_CONCEPT))

    # --- 【重要修正】 Plastic結合の設定 ---
    # Memory -> Motor への結合を学習可能(Plastic)に設定する
    # 初期値は0のままでも、Hebbian学習で重みが増加すればOK
    engine.mask_plastic[idx_m[0], idx_mem] = True

    # ---------------------------------------------------------
    # 3. シナリオ定義
    # ---------------------------------------------------------
    total_steps = 1500
    input_series = np.zeros((total_steps, N_SENSORY))
    
    def set_pulse(start, duration, channel):
        input_series[start:start+duration, channel] = 10.0

    # A. Pre-Test (Bell Only)
    set_pulse(100, 50, 0) 
    
    # B. Training (Bell -> Gap -> Food) x 5 trials
    train_start = 300
    interval = 150
    for i in range(5):
        t = train_start + i * interval
        set_pulse(t, 50, 0)      # Bell
        set_pulse(t + 60, 30, 1) # Food (Time lag: 60ms start diff)
        
    # C. Post-Test (Bell Only)
    test_start = train_start + 5 * interval + 100
    set_pulse(test_start, 50, 0)

    # ---------------------------------------------------------
    # 4. シミュレーション実行
    # ---------------------------------------------------------
    print(f"Running simulation for {total_steps} steps...")
    
    log_motor = []
    log_concept = []
    log_gate = []
    log_weights = []
    
    # 監視する重み (Memory -> Motor)
    monitor_plastic_indices = engine.mask_plastic[idx_m[0], :]
    
    for t in range(total_steps):
        current_input = np.zeros(engine.n_total)
        current_input[idx_s] = input_series[t]
        
        spikes = engine.step(current_input)
        
        log_motor.append(spikes[idx_m])
        log_concept.append(spikes[idx_c])
        log_gate.append(1.0 if engine.is_gating else 0.0)
        
        # 重みの平均値記録 (Plasticな部分のみ)
        w_plastic = engine.W[idx_m[0], monitor_plastic_indices]
        log_weights.append(np.mean(w_plastic) if len(w_plastic) > 0 else 0)

    # ---------------------------------------------------------
    # 5. 可視化 (Visualization)
    # ---------------------------------------------------------
    log_motor = np.array(log_motor)
    log_concept = np.array(log_concept)
    log_gate = np.array(log_gate)
    log_weights = np.array(log_weights)
    
    # --- CLI Report ---
    print_cli_heatmap(input_series[:, 0], title="Input: Bell (Pre-Test / Training / Post-Test)")
    print_cli_heatmap(input_series[:, 1], title="Input: Food (Training Only)")
    print_cli_heatmap(log_motor, title="Motor Output (Salivation)")

    print("\n[Result Check]")
    # Pre-Test区間の発火総数
    pre_test_response = np.sum(log_motor[100:200])
    # Post-Test区間の発火総数
    post_test_response = np.sum(log_motor[test_start:test_start+100])
    
    print(f"Pre-Test Response  (Bell Only): {pre_test_response}")
    print(f"Post-Test Response (Bell Only): {post_test_response}")
    
    if post_test_response > pre_test_response and post_test_response > 5:
        print("\n✅ SUCCESS: Conditioned Reflex Established! (Learning observed)")
    else:
        print("\n❌ FAILURE: No learning observed.")

    # --- Graph Plot ---
    # 保存先を reports/phase1_4_pavlov/ に変更
    output_dir = os.path.join(project_root, "reports/phase1_4_pavlov")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "result.png")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    
    # 1. Inputs
    axes[0].set_title("1. Sensory Input")
    axes[0].plot(input_series[:, 0], label="Bell", color='blue')
    axes[0].plot(input_series[:, 1], label="Food", color='orange', alpha=0.7)
    axes[0].legend(loc="upper right")
    
    # 2. Concept & Gate
    axes[1].set_title("2. Concept & Gate Signal")
    axes[1].plot(log_concept[:, 0], label="Concept: Bell", color='cyan', alpha=0.6)
    axes[1].fill_between(range(total_steps), 0, log_gate, color='red', alpha=0.1, label="Gate Open")
    axes[1].legend(loc="upper right")
    
    # 3. Motor
    axes[2].set_title("3. Motor Output (Salivation)")
    axes[2].plot(log_motor[:, 0], color='green', label="Salivation")
    axes[2].axvspan(100, 150, color='gray', alpha=0.2, label='Pre-Test')
    axes[2].axvspan(test_start, test_start+50, color='yellow', alpha=0.2, label='Post-Test')
    axes[2].legend()
    
    # 4. Weights
    axes[3].set_title("4. Mean Weight (Memory -> Motor)")
    axes[3].plot(log_weights, color='purple')
    axes[3].set_xlabel("Time (ms)")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    run_pavlov_experiment()