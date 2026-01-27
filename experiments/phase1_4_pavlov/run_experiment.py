import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(os.path.join(project_root, 'src'))

from core.engine import BiCortexEngine
from utils.cli_plotter import print_cli_heatmap, print_cli_float_series

def print_diagnostics(engine, log_mem, log_gate, log_weights, mem_bell_idx, mem_food_idx):
    """
    シミュレーション結果の診断レポートを出力する
    """
    print("\n" + "="*60)
    print("   DIAGNOSTIC REPORT")
    print("="*60)

    # 学習フェーズ (300-1050 step) のデータを抽出
    train_phase_slice = slice(300, 1050)
    
    # ゲート開閉率の確認
    gate_open_count = np.sum(log_gate[train_phase_slice])
    gate_ratio = gate_open_count / (1050 - 300)
    
    print(f"\n[SRG Gate]: Open Ratio = {gate_ratio:.2%} (Target: 20-50%)")
    
    # 重み変化の確認
    w_start = log_weights[0]
    w_end = log_weights[-1]
    w_delta = w_end - w_start
    
    print(f"\n[Weights]: Delta = {w_delta:.4f} (Max Clip: {engine.w_max_clip})")
    
    if w_delta > 0.5:
        print("  ✅ STATUS: STRONG ASSOCIATION LEARNED")
    elif w_delta > 0.01:
        print("  ✅ STATUS: MODERATE/EFFICIENT LEARNING (Success)")
    else:
        print("  ❌ STATUS: WEAK / NO LEARNING")
    print("="*60 + "\n")

def run_pavlov_experiment():
    print("=== Phase 1.4: Pavlov (Golden Parameters) ===")
    
    # 1. 設定
    N_SENSORY = 2 
    N_CONCEPT = 2 
    N_MOTOR   = 1 
    N_MEMORY  = 100 
    DT = 1.0 
    
    # エンジン初期化 (最適化済みのパラメータセット)
    engine = BiCortexEngine(
        n_sensory=N_SENSORY,
        n_concept=N_CONCEPT,
        n_motor=N_MOTOR,
        n_mem=N_MEMORY,
        dt=DT,
        
        # 学習率: 0.001 (過剰適合を防ぎ、必要最小限の結合を作る)
        learning_rate=0.001,
        
        # ゲート: 0.15 (不応期の影響を受けずに確実に開く閾値)
        gate_ratio=0.15,
        
        # 減衰: 0.005 (不要な記憶を消去する適度な強さ)
        global_decay=0.005,
        
        # 順応: 0.3 (刺激消失後のループを断ち切る絶妙なブレーキ)
        adaptation_step=0.3, 
        adaptation_tau=100.0,
        
        # 重み上限: 0.8 (リザーバ容量を圧迫しない範囲)
        w_max_clip=0.8,
        seed=42
    )
    
    # リザーバ初期化 & クリップ適用
    engine.init_memory_reservoir(density=0.2, spectral_radius=0.9)
    engine.W[engine.idx_mem, engine.idx_mem] = np.clip(
        engine.W[engine.idx_mem, engine.idx_mem], -1.0, 1.0
    )

    idx_s = engine.idx_sensory
    idx_c = engine.idx_concept
    idx_m = engine.idx_motor
    idx_mem = engine.idx_mem
    
    # 2. 配線 (Thinking Cortex & Interface)
    # 反射回路 (強)
    w_reflex = 8.0
    engine.W[idx_c[0], idx_s[0]] = w_reflex 
    engine.W[idx_c[1], idx_s[1]] = w_reflex 
    engine.W[idx_m[0], idx_c[1]] = w_reflex 
    
    mem_bell_indices = idx_mem[0:10]
    mem_food_indices = idx_mem[10:20]
    
    # Injection: 0.6 (確実に発火させるが、Traceを溜めすぎない)
    w_injection = 0.6
    engine.W[mem_bell_indices, idx_c[0]] = w_injection
    engine.W[mem_food_indices, idx_c[1]] = w_injection
    
    # Recall: 0.12 (数個の同期発火で概念を叩ける強度)
    w_recall = 0.12
    engine.W[idx_c[0], mem_bell_indices] = w_recall
    engine.W[idx_c[1], mem_food_indices] = w_recall
    
    # 3. 可塑性制御 (Memory->Memoryのみ学習)
    engine.mask_plastic[idx_c, :] = False 
    engine.mask_plastic[:, idx_c] = False 
    
    # 学習対象: Bell記憶 -> Food記憶
    grid_post_pre = np.ix_(mem_food_indices, mem_bell_indices)
    engine.W[grid_post_pre] = 0.0
    engine.mask_plastic[grid_post_pre] = True

    # 4. シナリオ作成
    total_steps = 1500
    input_series = np.zeros((total_steps, N_SENSORY))
    def set_pulse(start, duration, channel):
        input_series[start:start+duration, channel] = 10.0

    set_pulse(100, 50, 0) # Pre-Test (Bellのみ)
    
    # 学習セッション (5回)
    train_start = 300
    interval = 150
    for i in range(5):
        t = train_start + i * interval
        set_pulse(t, 50, 0)      # Bell
        set_pulse(t + 60, 30, 1) # Food
        
    set_pulse(train_start + 5 * interval + 100, 50, 0) # Post-Test (Bellのみ)

    # 5. シミュレーション実行
    print(f"Running simulation for {total_steps} steps...")
    
    log_motor = []
    log_concept = []
    log_gate = []
    log_mem = []
    log_weights_mean = []
    monitor_indices = grid_post_pre
    
    for t in range(total_steps):
        current_input = np.zeros(engine.n_total)
        current_input[idx_s] = input_series[t]
        spikes = engine.step(current_input)
        
        log_motor.append(spikes[idx_m])
        log_concept.append(spikes[idx_c])
        log_mem.append(spikes[idx_mem])
        log_gate.append(1.0 if engine.is_gating else 0.0)
        log_weights_mean.append(np.mean(engine.W[monitor_indices]))

    # 6. 可視化 & 評価
    log_motor = np.array(log_motor)
    log_concept = np.array(log_concept)
    log_mem = np.array(log_mem)
    log_gate = np.array(log_gate)
    log_weights_mean = np.array(log_weights_mean)
    
    print_diagnostics(engine, log_mem, log_gate, log_weights_mean, mem_bell_indices[0], mem_food_indices[0])

    print_cli_heatmap(input_series[:, 0], title="Input: Bell")
    print_cli_heatmap(log_concept[:, 1], title="Concept Output: Food")
    print_cli_heatmap(log_motor, title="Motor Output: Salivation")
    print_cli_float_series(log_weights_mean, title="Weight Evolution (Bell->Food)")

    print("\n[Result Check]")
    pre_test_response = np.sum(log_motor[100:200])
    post_test_response = np.sum(log_motor[-250:-150]) # Post-Test区間
    last_100_response = np.sum(log_motor[-100:])
    
    print(f"Pre-Test:  {pre_test_response}")
    print(f"Post-Test: {post_test_response}")
    print(f"Tail Check: {last_100_response}")
    
    if pre_test_response == 0 and post_test_response > 0:
        if last_100_response < post_test_response * 0.2: 
            print("\n✅ SUCCESS: Perfectly Balanced! Association formed and decayed.")
        else:
            print("\n⚠️ WARNING: Learned well, but decay is slow.")
    else:
        print("\n❌ FAILURE: Learning failed.")

    # グラフ保存
    output_dir = os.path.join(project_root, "reports/phase1_4_pavlov")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "result.png")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(input_series[:, 0], label="Bell", color='blue')
    axes[0].plot(input_series[:, 1], label="Food", color='orange', alpha=0.7)
    axes[0].legend()
    axes[0].set_title("Input")
    
    axes[1].plot(log_concept[:, 1], color='magenta', label="Food Concept")
    axes[1].fill_between(range(total_steps), 0, log_gate, color='red', alpha=0.1, label="Gate")
    axes[1].legend()
    axes[1].set_title("Concept & Gate")
    
    axes[2].plot(log_motor[:, 0], color='green', label="Salivation")
    axes[2].set_title("Motor Output")
    
    axes[3].plot(log_weights_mean, color='purple', label="Weights")
    axes[3].set_title("Weight Evolution")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    run_pavlov_experiment()