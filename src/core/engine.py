import numpy as np

class BiCortexEngine:
    """
    Bi-Cortex SNN Engine (Associative Memory Model)
    
    Phase 1.4 Architecture:
    - Thinking Cortex (Fixed): 感覚(Sensory) -> 概念(Concept) -> 運動(Motor)
    - Interface (Fixed): 概念と記憶野を結ぶ固定結合 (Injection / Recall)
    - Memory Cortex (Plastic): リザーバ計算を行う記憶野。SRGにより内部結合のみ学習する。
    """

    def __init__(self, 
                 n_sensory: int, 
                 n_concept: int, 
                 n_motor: int, 
                 n_mem: int, 
                 dt: float = 1.0, 
                 learning_rate: float = 0.05,
                 gate_ratio: float = 0.1, 
                 global_decay: float = 0.001,
                 adaptation_step: float = 0.0, 
                 adaptation_tau: float = 100.0,
                 refractory_period: float = 2.0,
                 w_max_clip: float = 5.0,
                 seed: int = 42):
        """
        エンジンの初期化

        Args:
            n_sensory, n_concept, n_motor, n_mem (int): 各領域のニューロン数
            dt (float): シミュレーションのタイムステップ (ms)
            learning_rate (float): SRG学習率 (eta)。Trace蓄積を考慮し低めに設定すること。
            gate_ratio (float): 学習ゲートを開く活動閾値の割合 (0.0~1.0)。
            global_decay (float): 重み減衰率。記憶の固定化と消去のバランス調整用。
            adaptation_step (float): 順応強度。発火ごとの閾値上昇量。無限ループ防止用。
            adaptation_tau (float): 順応の回復時定数 (ms)。
            refractory_period (float): 不応期 (ms)。
            w_max_clip (float): 重みのクリッピング上限（絶対値）。
            seed (int): 乱数シード。
        """
        
        self.rng = np.random.default_rng(seed)
        
        # --- 1. 領域定義 (Parcellation) ---
        self.n_sensory = n_sensory
        self.n_concept = n_concept
        self.n_motor = n_motor
        self.n_mem = n_mem
        
        self.n_think = n_sensory + n_concept + n_motor
        self.n_total = self.n_think + n_mem
        
        # ID Ranges
        self.idx_sensory = np.arange(0, n_sensory)
        self.idx_concept = np.arange(n_sensory, n_sensory + n_concept)
        self.idx_motor = np.arange(n_sensory + n_concept, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        # Memory E/I Balance (Dale's Law: 80% Exc, 20% Inh)
        self.n_exc = int(n_mem * 0.8)
        self.n_inh = n_mem - self.n_exc
        self.mem_exc_mask = np.zeros(n_mem, dtype=bool)
        self.mem_exc_mask[:self.n_exc] = True
        self.mem_inh_mask = ~self.mem_exc_mask
        
        # --- 2. ハイパーパラメータ ---
        self.dt = dt
        self.learning_rate = learning_rate
        self.global_decay = global_decay
        self.w_max_clip = w_max_clip
        
        # Gate Threshold: 不応期による発火率制限(Max 0.5)を考慮し、最小値を0.1に設定
        self.gate_threshold = max(0.1, n_concept * gate_ratio)
        
        # SRG State
        self.activity_ma = 0.0
        self.ma_alpha = 0.2
        self.is_gating = False

        # --- 3. ニューロン状態変数 (LIF) ---
        self.v = np.zeros(self.n_total)
        self.v_base = 5.0 
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(refractory_period / dt))

        # Adaptation (順応)
        self.adaptation = np.zeros(self.n_total)
        self.adaptation_step = adaptation_step
        self.decay_adapt = np.exp(-dt / adaptation_tau)

        # --- 4. Trace変数 (Synaptic Eligibility) ---
        # Long-term Trace (学習用)
        self.e_trace = np.zeros(self.n_total)
        self.tau_trace = 2000.0 
        self.decay_trace = np.exp(-dt / self.tau_trace)

        # Short-term Trace (信号伝達用)
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 5.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        # --- 5. 結合行列 ---
        self.W = np.zeros((self.n_total, self.n_total))
        # 可塑性マスク: Trueの箇所のみSRGで更新される
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)

    def init_memory_reservoir(self, density=0.1, spectral_radius=0.9):
        """
        記憶野をリザーバ（Echo State Network状）として初期化する。
        Dale's Lawに基づき、興奮性ニューロンからは正、抑制性からは負の結合を出力する。
        """
        W_mem = np.zeros((self.n_mem, self.n_mem))
        mask = self.rng.random((self.n_mem, self.n_mem)) < density
        weights = self.rng.random((self.n_mem, self.n_mem))
        
        # E/I 特性の適用
        W_mem[:, self.mem_exc_mask] = weights[:, self.mem_exc_mask]   # Exc -> +
        W_mem[:, self.mem_inh_mask] = -weights[:, self.mem_inh_mask]  # Inh -> -
        
        # スペクトル半径の調整
        W_mem *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W_mem)))
        if radius > 0:
            W_mem *= (spectral_radius / radius)
            
        # 初期重みに対してもクリッピングを適用 (重要)
        W_mem = np.clip(W_mem, -self.w_max_clip, self.w_max_clip)

        # 全体結合行列へ適用 & 可塑性フラグの設定
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = W_mem
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = (W_mem != 0)

    def reset_state(self):
        """状態変数のリセット（重みは保持）"""
        self.v[:] = 0
        self.x_fast[:] = 0
        self.e_trace[:] = 0
        self.refractory_count[:] = 0
        self.adaptation[:] = 0 
        self.activity_ma = 0.0

    def step(self, input_current: np.ndarray):
        """
        1タイムステップのシミュレーションを実行する
        Sequence: Integration -> Fire -> Adaptation -> Trace -> Gating -> Learning
        """
        # 1. 膜電位の更新 (LIF)
        synaptic_input = self.W @ self.x_fast
        self.v = self.v * self.alpha + input_current + synaptic_input
        
        # 2. 順応の減衰と閾値決定
        self.adaptation *= self.decay_adapt 
        v_thresh = self.v_base + self.adaptation
        
        # 不応期中のニューロンは強制リセット
        self.v[self.refractory_count > 0] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        # 3. 発火判定
        spikes = (self.v >= v_thresh).astype(float)
        fired = np.where(spikes > 0)[0]
        
        # 発火後処理: リセット、不応期設定、順応加算
        self.v[fired] = 0.0
        self.refractory_count[fired] = self.refractory_steps
        if self.adaptation_step > 0:
            self.adaptation[fired] += self.adaptation_step
        
        # 4. トレース変数の更新
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_trace = self.e_trace * self.decay_trace + spikes
        
        # 5. SRGゲート判定 (Thinking CortexのConcept活動を監視)
        concept_activity = np.sum(spikes[self.idx_concept])
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        self.is_gating = (self.activity_ma >= self.gate_threshold)

        # 6. 可塑性更新
        self._update_weights_srg(spikes)

        return spikes

    def _update_weights_srg(self, spikes: np.ndarray):
        """
        Semantic Resonance Gating による重み更新
        Rule: Delta W = eta * Gate * Post(t) * Pre_Trace(t)
        """
        # A. 全体減衰 (Global Decay) - 忘却プロセス
        if self.global_decay > 0:
             self.W[self.mask_plastic] *= (1.0 - self.global_decay)
        
        # ゲートが閉じている場合は学習しない
        if not self.is_gating:
            return

        # B. ヘブ則更新 (Gated Hebbian)
        fired = np.where(spikes > 0)[0]
        
        for post in fired:
            # このニューロンへの入力結合のうち、可塑性があるものを抽出
            mask_p = self.mask_plastic[post, :]
            if not np.any(mask_p): continue
            
            # Pre-synaptic Activity (Trace)
            pre_trace = self.e_trace[mask_p]
            
            # 更新量の計算
            delta = self.learning_rate * pre_trace
            
            # 重み更新とクリッピング
            w = self.W[post, mask_p]
            new_w = w + delta
            self.W[post, mask_p] = np.clip(new_w, -self.w_max_clip, self.w_max_clip)