import numpy as np

class BiCortexEngine:
    def __init__(self, 
                 n_input: int,   # TC (Sensory)
                 n_hidden: int,  # TC (Internal)
                 n_motor: int,   # TC (Motor)
                 n_mem: int,     # Memory Cortex
                 dt: float = 1.0, 
                 learning_rate: float = 0.05,
                 gate_ratio: float = 0.02,
                 global_decay: float = 0.02, # 忘却を強めに（誤学習を即消去）
                 w_scale_fixed: float = 0.0, 
                 w_scale_rec: float = 0.05,  
                 adaptation_beta: float = 0.0, 
                 adaptation_tau: float = 100.0,
                 seed: int = 42):
        """
        Bi-Cortex SNN Engine (v10.1: High-Contrast Tuning)
        """
        self.rng = np.random.default_rng(seed)
        
        # 1. 領域定義
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_motor = n_motor
        self.n_think = n_input + n_hidden + n_motor
        self.n_mem = n_mem
        self.n_total = self.n_think + n_mem
        self.dt = dt
        
        # 2. パラメータ
        self.learning_rate = learning_rate
        self.gate_threshold = max(0.1, self.n_input * gate_ratio)
        self.global_decay = global_decay
        
        self.w_scale_fixed = w_scale_fixed
        self.w_scale_rec = w_scale_rec
        
        self.adaptation_beta = adaptation_beta
        self.adaptation_decay = np.exp(-dt / adaptation_tau)

        # 3. IDマップ
        self.idx_input = np.arange(0, n_input)
        self.idx_hidden = np.arange(n_input, n_input + n_hidden)
        self.idx_motor = np.arange(n_input + n_hidden, self.n_think)
        self.idx_think = np.arange(0, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        self.activity_ma = 0.0
        self.ma_alpha = 0.2
        self.last_gate_status = 0.0

        # 4. ニューロン状態 (LIF)
        self.v = np.zeros(self.n_total)
        # 閾値を高めに設定し、ノイズ耐性を上げる
        self.v_base = 2.5 
        self.v_th_adaptive = np.zeros(self.n_total)
        
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(2.0 / dt))

        # 5. トレース変数 (Dual Traces)
        # x_fast (PSC)
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 5.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        # e_slow (Eligibility Trace)
        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0
        self.decay_slow = np.exp(-dt / self.tau_slow)

        # 6. 結合行列 (Interface & Wiring)
        self.W = np.zeros((self.n_total, self.n_total))
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)
        
        self._build_topology()

    def _build_topology(self):
        # A. Thinking Internal (Fixed)
        if self.n_hidden > 0 and self.w_scale_fixed > 0:
            self.W[np.ix_(self.idx_hidden, self.idx_input)] = \
                self.rng.random((self.n_hidden, self.n_input)) * self.w_scale_fixed
            self.W[np.ix_(self.idx_motor, self.idx_hidden)] = \
                self.rng.random((self.n_motor, self.n_hidden)) * self.w_scale_fixed

        # B. Memory Internal (Plastic)
        if self.w_scale_rec > 0:
            w_rec = self.rng.random((self.n_mem, self.n_mem)) * self.w_scale_rec
            np.fill_diagonal(w_rec, 0.0)
            self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec
            self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

    def reset_state(self):
        """
        トライアル間の完全リセット。
        """
        self.v[:] = 0
        self.x_fast[:] = 0
        self.e_slow[:] = 0
        self.refractory_count[:] = 0

    def step(self, input_current: np.ndarray):
        # Synaptic Input
        synaptic_input = self.W @ self.x_fast
        
        # LIF Dynamics
        is_refractory = self.refractory_count > 0
        current_threshold = self.v_base + self.v_th_adaptive
        
        self.v = self.v * self.alpha + input_current + synaptic_input
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        # Fire
        spikes = (self.v >= current_threshold).astype(float)
        fired = np.where(spikes > 0)[0]
        
        # Reset
        self.v[fired] = 0.0
        self.refractory_count[fired] = self.refractory_steps
        
        self.v_th_adaptive[fired] += self.adaptation_beta
        self.v_th_adaptive *= self.adaptation_decay

        # Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes
        self.e_slow = np.clip(self.e_slow, 0.0, 5.0)

        return spikes

    def update_weights(self, spikes: np.ndarray, reward: float):
        # 1. Decay (報酬がないと忘れる)
        if self.global_decay > 0:
             self.W[self.mask_plastic] *= (1.0 - self.global_decay)

        if reward == 0.0: return

        # 2. Reinforcement (報酬があれば強める)
        fired = np.where(spikes > 0)[0]
        pre_trace = self.e_slow
        W_MAX = 5.0; W_MIN = -5.0
        
        targets = fired
        if abs(reward) > 0.01:
            active_trace_indices = np.where(self.e_slow > 0.1)[0]
            targets = np.unique(np.concatenate([fired, active_trace_indices]))

        for post in targets:
            mask = self.mask_plastic[post, :]
            if not np.any(mask): continue
            
            w = self.W[post, mask]
            pre = pre_trace[mask]
            
            delta = self.learning_rate * reward * pre
            
            new_w = w + delta
            self.W[post, mask] = np.clip(new_w, W_MIN, W_MAX)