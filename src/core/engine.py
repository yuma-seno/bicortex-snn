import numpy as np

class BiCortexEngine:
    def __init__(self, 
                 n_input: int, 
                 n_hidden: int, 
                 n_motor: int, 
                 n_mem: int, 
                 dt: float = 1.0, 
                 learning_rate: float = 0.005,
                 gate_ratio: float = 0.05,      # 【変更】閾値(絶対値) -> 閾値割合(Ratio)
                 global_decay: float = 0.00001, # 【追加】全体忘却率
                 seed: int = 42):
        """
        Bi-Cortex SNN Core Engine v1.2 (Stable)
        """
        self.rng = np.random.default_rng(seed)
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_motor = n_motor
        self.n_think = n_input + n_hidden + n_motor
        self.n_mem = n_mem
        self.n_total = self.n_think + n_mem
        self.dt = dt
        
        # Parameters
        self.learning_rate = learning_rate
        # 【修正3】閾値を動的に計算 (思考野のニューロン数 * 割合)
        # 例: 100ニューロン * 0.05 = 5ニューロン以上活動でゲートOPEN
        self.gate_threshold = self.n_think * gate_ratio
        
        # 【修正2】全体減衰率 (使われない結合を忘却させる)
        self.global_decay = global_decay 

        # --- 1. ID Mapping ---
        self.idx_input = np.arange(0, n_input)
        self.idx_hidden = np.arange(n_input, n_input + n_hidden)
        self.idx_motor = np.arange(n_input + n_hidden, self.n_think)
        self.idx_think = np.arange(0, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        # Gate Activity Monitoring
        self.activity_ma = 0.0
        self.ma_alpha = 0.1

        # --- 2. Neuron States ---
        self.v = np.zeros(self.n_total)
        self.v_th = 1.0
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        
        # 【修正1】不応期が0にならないよう安全策
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(2.0 / dt))

        # --- 3. Dual Traces ---
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 20.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0
        self.decay_slow = np.exp(-dt / self.tau_slow)

        # --- 4. Weights ---
        self.W = np.zeros((self.n_total, self.n_total))
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)
        
        self._build_topology()

    def _build_topology(self):
        """結合トポロジー構築 (v1.2)"""
        n_inh = int(self.n_mem * 0.2)
        self.idx_mem_inh = self.idx_mem[-n_inh:] if n_inh > 0 else np.array([], dtype=int)
        self.idx_mem_exc = np.setdiff1d(self.idx_mem, self.idx_mem_inh)

        # 【修正6】初期重みのスケールを全体的に下げる (*1.5 -> *0.8など)
        # 入力が多数重なっても即発火しないように抑制
        scale_fixed = 0.8
        scale_plastic = 0.5

        # --- A. Thinking Cortex Internal (Fixed) ---
        if self.n_hidden > 0:
            w_i2h = self.rng.random((self.n_hidden, self.n_input)) * scale_fixed
            mask_i2h = self.rng.random((self.n_hidden, self.n_input)) < 0.3
            self.W[np.ix_(self.idx_hidden, self.idx_input)] = w_i2h * mask_i2h

            w_h2m = self.rng.random((self.n_motor, self.n_hidden)) * scale_fixed
            mask_h2m = self.rng.random((self.n_motor, self.n_hidden)) < 0.3
            self.W[np.ix_(self.idx_motor, self.idx_hidden)] = w_h2m * mask_h2m
        else:
            w_i2m = self.rng.random((self.n_motor, self.n_input)) * scale_fixed
            mask_i2m = self.rng.random((self.n_motor, self.n_input)) < 0.3
            self.W[np.ix_(self.idx_motor, self.idx_input)] = w_i2m * mask_i2m

        # --- B. Thinking -> Memory (Fixed) ---
        src_indices = np.concatenate([self.idx_input, self.idx_hidden])
        w_t2m = self.rng.random((self.n_mem, len(src_indices))) * 0.5
        mask_t2m = self.rng.random((self.n_mem, len(src_indices))) < 0.1
        self.W[np.ix_(self.idx_mem, src_indices)] = w_t2m * mask_t2m

        # --- C. Memory -> Memory (Plastic) ---
        w_rec = self.rng.random((self.n_mem, self.n_mem)) * scale_plastic
        np.fill_diagonal(w_rec, 0.0)
        mask_rec = self.rng.random((self.n_mem, self.n_mem)) < 0.1
        np.fill_diagonal(mask_rec, False)
        
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec * mask_rec
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # --- D. Memory -> Thinking (Plastic) ---
        if self.n_hidden > 0:
             self.mask_plastic[np.ix_(self.idx_hidden, self.idx_mem)] = True
        self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = True

        # --- E. Apply Dale's Law ---
        if len(self.idx_mem_inh) > 0:
            self.W[:, self.idx_mem_inh] *= -1.0
            self.W[:, self.idx_mem_inh] *= 2.0

    def step(self, input_current: np.ndarray, learning: bool = True):
        if input_current.shape[0] != self.n_total:
            raise ValueError(f"Input mismatch: {input_current.shape[0]} != {self.n_total}")

        # 1. LIF Update
        synaptic_input = self.W @ self.x_fast
        
        is_refractory = self.refractory_count > 0
        self.v = self.v * self.alpha + input_current + synaptic_input
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        spikes = (self.v >= self.v_th).astype(float)
        
        fired_indices = np.where(spikes > 0)[0]
        self.v[fired_indices] = 0.0
        self.refractory_count[fired_indices] = self.refractory_steps

        # 2. Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes

        # 3. Learning (SRG)
        if learning:
            self._update_weights(spikes)
            
        # 【修正2】全体減衰 (Global Weight Decay)
        # 学習対象(Plastic)の重みだけを少しずつ減衰させる
        # W_new = W_old * (1 - decay)
        if self.global_decay > 0:
            self.W[self.mask_plastic] *= (1.0 - self.global_decay)

        return spikes

    def _update_weights(self, spikes):
        """SRG Rule with Soft-bound"""
        # A. ゲート信号 (移動平均 vs 相対閾値)
        concept_activity = np.sum(spikes[self.idx_input])
        if self.n_hidden > 0:
            concept_activity += np.sum(spikes[self.idx_hidden])
            
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        
        G_t = 1.0 if self.activity_ma >= self.gate_threshold else 0.0
        
        if G_t == 0.0:
            return

        # B. 重み更新
        fired_indices = np.where(spikes > 0)[0]
        if len(fired_indices) == 0:
            return

        pre_trace = self.e_slow
        W_MAX_EXC = 2.0
        W_MIN_INH = -2.0

        for post_idx in fired_indices:
            target_mask = self.mask_plastic[post_idx, :]
            if not np.any(target_mask):
                continue

            current_W = self.W[post_idx, target_mask]
            pre_indices = np.where(target_mask)[0]
            
            is_inh = np.isin(pre_indices, self.idx_mem_inh)
            is_exc = ~is_inh
            
            delta_w = np.zeros_like(current_W)
            
            # 1. Excitatory -> Potentiation
            if np.any(is_exc):
                delta_w[is_exc] = (W_MAX_EXC - current_W[is_exc]) * \
                                  self.learning_rate * G_t * pre_trace[pre_indices[is_exc]]

            # 2. Inhibitory -> Potentiation (More negative)
            if np.any(is_inh):
                delta_w[is_inh] = -1.0 * (current_W[is_inh] - W_MIN_INH) * \
                                  self.learning_rate * G_t * pre_trace[pre_indices[is_inh]]

            # ※ 個別減衰(Weight Decay)は Global Decay に任せるためここでは削除

            self.W[post_idx, target_mask] += delta_w

        # C. Safety Clipping
        col_exc = np.setdiff1d(np.arange(self.n_total), self.idx_mem_inh)
        self.W[:, col_exc] = np.clip(self.W[:, col_exc], 0.0, W_MAX_EXC)
        
        if len(self.idx_mem_inh) > 0:
            self.W[:, self.idx_mem_inh] = np.clip(self.W[:, self.idx_mem_inh], W_MIN_INH, 0.0)