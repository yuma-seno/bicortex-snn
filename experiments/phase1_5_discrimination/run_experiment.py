import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# パス設定 (プロジェクトルートのモジュールを読み込めるようにする)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(os.path.join(project_root, 'src'))

from core.engine import BiCortexEngine
from utils.cli_plotter import print_cli_heatmap, print_cli_float_series

def run_discrimination_experiment():
    print("=== Phase 1.5: Discrimination Task (Red=Reward, Blue=None) ===")
    
    # 1. 設定定義
    # Sensory: 0=Red, 1=Blue, 2=Reward
    # Concept: 0=Red, 1=Blue, 2=Reward
    N_SENSORY = 3 
    N_CONCEPT = 3 
    N_MOTOR   = 1 
    N_MEMORY  = 100 
    DT = 1.0 
    
    # エンジン初期化 (Phase 1.4 Golden Parameters)
    engine = BiCortexEngine(
        n_sensory=N_SENSORY,
        n_concept=N_CONCEPT,
        n_motor=N_MOTOR,
        n_mem=N_MEMORY,
        dt=DT,
        learning_rate=0.001,      # 低学習率で慎重に結合
        gate_ratio=0.15,          # ゲート感度
        global_decay=0.005,       # 忘却率
        adaptation_step=0.3,      # 順応ブレーキ (重要)
        adaptation_tau=100.0,
        w_max_clip=0.8,
        seed=42
    )
    
    # リザーバ初期化
    engine.init_memory_reservoir(density=0.2, spectral_radius=0.9)
    # 重みクリップ
    engine.W[engine.idx_mem, engine.idx_mem] = np.clip(
        engine.W[engine.idx_mem, engine.idx_mem], -1.0, 1.0
    )

    # インデックス短縮名
    idx_s = engine.idx_sensory
    idx_c = engine.idx_concept
    idx_m = engine.idx_motor
    idx_mem = engine.idx_mem
    
    # 2. 思考野 (Thinking Cortex) の配線構築
    w_reflex = 8.0
    # 感覚 -> 概念 (1対1対応)
    engine.W[idx_c[0], idx_s[0]] = w_reflex  # Red -> Red Concept
    engine.W[idx_c[1], idx_s[1]] = w_reflex  # Blue -> Blue Concept
    engine.W[idx_c[2], idx_s[2]] = w_reflex  # Reward -> Reward Concept
    
    # 概念 -> 運動 (報酬概念のみが行動を誘発)
    engine.W[idx_m[0], idx_c[2]] = w_reflex  # Reward Concept -> Action
    
    # 3. インターフェース (Interface) の配線構築
    # 記憶野を3つの領域に仮想的に分割して割り当て（オーバーラップも許容するが今回は分離）
    mem_red_indices    = idx_mem[0:20]
    mem_blue_indices   = idx_mem[20:40]
    mem_reward_indices = idx_mem[40:60]
    
    w_injection = 0.6
    w_recall = 0.12
    
    # Injection: 概念 -> 記憶
    engine.W[mem_red_indices, idx_c[0]]    = w_injection
    engine.W[mem_blue_indices, idx_c[1]]   = w_injection
    engine.W[mem_reward_indices, idx_c[2]] = w_injection
    
    # Recall: 記憶 -> 概念
    engine.W[idx_c[0], mem_red_indices]    = w_recall
    engine.W[idx_c[1], mem_blue_indices]   = w_recall
    engine.W[idx_c[2], mem_reward_indices] = w_recall
    
    # 4. 可塑性制御 (SRG対象の設定)
    # 思考野やインターフェースは固定
    engine.mask_plastic[idx_c, :] = False 
    engine.mask_plastic[:, idx_c] = False 
    
    # 学習対象: 「Red/Blueの記憶」から「Rewardの記憶」への結合のみ許可
    # これにより Red->Reward, Blue->Reward の可能性が開かれるが、
    # 実際に強化されるのは同時発火した方のみ。
    grid_post_pre = np.ix_(mem_reward_indices, np.concatenate([mem_red_indices, mem_blue_indices]))
    engine.W[grid_post_pre] = 0.0
    engine.mask_plastic[grid_post_pre] = True

    # 5. シナリオ作成
    total_steps = 2500
    input_series = np.zeros((total_steps, N_SENSORY))
    
    def set_trial(start_time, stimulus_idx, has_reward):
        # 刺激呈示 (50step)
        input_series[start_time : start_time+50, stimulus_idx] = 10.0
        # 報酬呈示 (遅延60step後, 30step間)
        if has_reward:
            input_series[start_time+60 : start_time+90, 2] = 10.0

    # --- シナリオ構成 ---
    # 0-300: Pre-Test (学習前)
    set_trial(100, 0, False) # Red only
    set_trial(200, 1, False) # Blue only
    
    # 300-2000: Training (ランダム学習)
    # 赤は報酬あり、青は報酬なしを交互に近い形で繰り返す
    rng = np.random.default_rng(42)
    train_start = 400
    interval = 180
    
    for i in range(8):
        t = train_start + i * interval
        if i % 2 == 0:
            # Red -> Reward Pair
            set_trial(t, 0, True)
        else:
            # Blue -> No Reward
            set_trial(t, 1, False)
            
    # 2000-: Post-Test (学習後)
    test_start = train_start + 8 * interval + 100
    set_trial(test_start, 0, False)      # Red Test (Expect Reaction)
    set_trial(test_start + 200, 1, False) # Blue Test (Expect No Reaction)

    # 6. シミュレーション実行
    print(f"Running simulation for {total_steps} steps...")
    
    log_motor = []
    log_concept = []
    log_weights_red_rew = []
    log_weights_blue_rew = []
    
    # モニタリング用インデックス
    idx_red_to_rew = np.ix_(mem_reward_indices, mem_red_indices)
    idx_blue_to_rew = np.ix_(mem_reward_indices, mem_blue_indices)
    
    for t in range(total_steps):
        current_input = np.zeros(engine.n_total)
        current_input[idx_s] = input_series[t]
        
        spikes = engine.step(current_input)
        
        log_motor.append(spikes[idx_m])
        log_concept.append(spikes[idx_c])
        
        # 重みの平均値を記録
        log_weights_red_rew.append(np.mean(engine.W[idx_red_to_rew]))
        log_weights_blue_rew.append(np.mean(engine.W[idx_blue_to_rew]))

    # 7. 結果評価 & 可視化
    log_motor = np.array(log_motor)
    log_concept = np.array(log_concept)
    log_weights_red_rew = np.array(log_weights_red_rew)
    log_weights_blue_rew = np.array(log_weights_blue_rew)
    
    # CLI Report
    print_cli_heatmap(input_series[:, 0], title="Input: Red Stimulus")
    print_cli_heatmap(input_series[:, 1], title="Input: Blue Stimulus")
    print_cli_heatmap(input_series[:, 2], title="Input: Reward")
    print_cli_heatmap(log_motor, title="Motor Output: Action")
    
    print("\n[Weight Evolution]")
    print(f"Red->Reward Weight Delta: {log_weights_red_rew[-1] - log_weights_red_rew[0]:.4f}")
    print(f"Blue->Reward Weight Delta: {log_weights_blue_rew[-1] - log_weights_blue_rew[0]:.4f}")
    
    # 成功判定
    # Post-Testでの反応量を確認
    # Red Test区間: test_start ~ test_start+100
    # Blue Test区間: test_start+200 ~ test_start+300
    red_test_response = np.sum(log_motor[test_start : test_start+100])
    blue_test_response = np.sum(log_motor[test_start+200 : test_start+300])
    
    print(f"\n[Test Results]")
    print(f"Red Stimulus Response:  {red_test_response}")
    print(f"Blue Stimulus Response: {blue_test_response}")
    
    if red_test_response > 5 and blue_test_response < red_test_response * 0.2:
        print("\n✅ SUCCESS: Discrimination Learned! (Red triggers action, Blue does not)")
    elif red_test_response > 5 and blue_test_response > 5:
        print("\n⚠️ PARTIAL: Generalized too much (Reacts to both)")
    else:
        print("\n❌ FAILURE: Failed to learn association")

    # グラフ保存
    output_dir = os.path.join(project_root, "reports/phase1_5_discrimination")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "result_discrimination.png")
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    
    # Input
    axes[0].plot(input_series[:, 0], label="Red", color='red')
    axes[0].plot(input_series[:, 1], label="Blue", color='blue')
    axes[0].plot(input_series[:, 2], label="Reward", color='orange', linestyle='--')
    axes[0].legend(loc='upper right')
    axes[0].set_title("Sensory Inputs")
    
    # Concept
    axes[1].plot(log_concept[:, 0], label="Concept: Red", color='red', alpha=0.6)
    axes[1].plot(log_concept[:, 1], label="Concept: Blue", color='blue', alpha=0.6)
    axes[1].plot(log_concept[:, 2], label="Concept: Reward", color='orange', alpha=0.8)
    axes[1].legend(loc='upper right')
    axes[1].set_title("Thinking Cortex Concepts")
    
    # Motor
    axes[2].plot(log_motor[:, 0], color='green', label="Action")
    axes[2].set_title("Motor Output")
    
    # Weights
    axes[3].plot(log_weights_red_rew, label="Red->Reward Weights", color='red')
    axes[3].plot(log_weights_blue_rew, label="Blue->Reward Weights", color='blue')
    axes[3].legend(loc='upper left')
    axes[3].set_title("Memory Plasticity Evolution")
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")

if __name__ == "__main__":
    run_discrimination_experiment()