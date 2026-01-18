import numpy as np

class BiCortexEngine:
    def __init__(self, 
                 n_input: int, 
                 n_hidden: int, 
                 n_motor: int, 
                 n_mem: int, 
                 dt: float = 1.0, 
                 learning_rate: float = 0.005,
                 gate_ratio: float = 0.05,
                 global_decay: float = 0.01, # 少しだけ忘却を入れる(不要な結合を消すため)
                 w_scale_fixed: float = 0.0,
                 w_scale_rec: float = 0.05,
                 adaptation_beta: float = 0.5, # 疲れやすさ
                 adaptation_tau: float = 50.0, # 回復速度
                 seed: int = 42):
        """
        Bi-Cortex SNN Core Engine v2.3 (Plastic Core / Fixed Interface)
        """
        self.rng = np.random.default_rng(seed)
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_motor = n_motor
        self.n_think = n_input + n_hidden + n_motor
        self.n_mem = n_mem
        self.n_total = self.n_think + n_mem
        self.dt = dt
        
        self.learning_rate = learning_rate
        self.gate_threshold = max(0.1, self.n_input * gate_ratio)
        self.global_decay = global_decay
        
        self.w_scale_fixed = w_scale_fixed
        self.w_scale_rec = w_scale_rec
        
        self.adaptation_beta = adaptation_beta
        self.adaptation_decay = np.exp(-dt / adaptation_tau)

        self.idx_input = np.arange(0, n_input)
        self.idx_hidden = np.arange(n_input, n_input + n_hidden)
        self.idx_motor = np.arange(n_input + n_hidden, self.n_think)
        self.idx_think = np.arange(0, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        self.activity_ma = 0.0
        self.ma_alpha = 0.2
        self.last_gate_status = 0.0

        self.v = np.zeros(self.n_total)
        self.v_base = 1.0
        self.v_th_adaptive = np.zeros(self.n_total)
        
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(2.0 / dt))

        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 20.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0
        self.decay_slow = np.exp(-dt / self.tau_slow)

        self.W = np.zeros((self.n_total, self.n_total))
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)
        
        self._build_topology()

    def _build_topology(self):
        # A. Thinking Internal (Fixed if used)
        if self.n_hidden > 0 and self.w_scale_fixed > 0:
            self.W[np.ix_(self.idx_hidden, self.idx_input)] = \
                self.rng.random((self.n_hidden, self.n_input)) * self.w_scale_fixed
            self.W[np.ix_(self.idx_motor, self.idx_hidden)] = \
                self.rng.random((self.n_motor, self.n_hidden)) * self.w_scale_fixed

        # B. Context Injection (Thinking -> Memory) [FIXED]
        if self.w_scale_fixed > 0:
            src = np.concatenate([self.idx_input, self.idx_hidden])
            self.W[np.ix_(self.idx_mem, src)] = \
                self.rng.random((self.n_mem, len(src))) * self.w_scale_fixed

        # C. Recurrent (Memory <-> Memory) [PLASTIC]
        # ★ここがコンセプトの核: 記憶野内部の配線のみを学習させる
        if self.w_scale_rec > 0:
            w_rec = self.rng.random((self.n_mem, self.n_mem)) * self.w_scale_rec
            np.fill_diagonal(w_rec, 0.0)
            self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec
            
            # 学習許可: 記憶野内部のみTrue
            self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # D. Interface (Memory -> Motor) [FIXED]
        # ★出力への配線は固定する（学習しない）
        # self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = False # Default False

        # E. Dale's Law (Inhibition)
        n_inh = int(self.n_mem * 0.2)
        if n_inh > 0:
            self.idx_mem_inh = self.idx_mem[-n_inh:]
            self.W[:, self.idx_mem_inh] *= -1.0
            self.W[:, self.idx_mem_inh] *= 2.0
            # 抑制性ニューロンからの出力は（通常は）学習させないことが多いが
            # 今回はシンプルにExc->Excの強化をメインにするため、そのままでOK
        else:
            self.idx_mem_inh = np.array([], dtype=int)

    def step(self, input_current: np.ndarray, learning: bool = True):
        # 1. Synaptic Input
        synaptic_input = self.W @ self.x_fast
        
        # 2. LIF with Adaptation
        is_refractory = self.refractory_count > 0
        current_threshold = self.v_base + self.v_th_adaptive
        
        self.v = self.v * self.alpha + input_current + synaptic_input
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        spikes = (self.v >= current_threshold).astype(float)
        fired = np.where(spikes > 0)[0]
        
        self.v[fired] = 0.0
        self.refractory_count[fired] = self.refractory_steps
        
        # Adaptation
        self.v_th_adaptive[fired] += self.adaptation_beta
        self.v_th_adaptive *= self.adaptation_decay

        # 3. Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes

        # 4. Learning
        if learning:
            self._update_weights(spikes)
            
        if self.global_decay > 0:
            self.W[self.mask_plastic] *= (1.0 - self.global_decay)

        return spikes

    def _update_weights(self, spikes):
        concept_activity = np.sum(spikes[self.idx_input])
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        
        G_t = 1.0 if self.activity_ma >= self.gate_threshold else 0.0
        self.last_gate_status = G_t

        if G_t == 0.0: return

        fired = np.where(spikes > 0)[0]
        if len(fired) == 0: return

        pre_trace = self.e_slow
        W_MAX = 2.0; W_MIN = -2.0

        for post in fired:
            mask = self.mask_plastic[post, :]
            if not np.any(mask): continue
            
            w = self.W[post, mask]
            delta = self.learning_rate * G_t * pre_trace[mask]
            
            # Soft-bound
            is_pos = w >= 0
            if np.any(is_pos):
                w[is_pos] += delta[is_pos] * (W_MAX - w[is_pos])
            is_neg = w < 0
            if np.any(is_neg):
                w[is_neg] -= delta[is_neg] * (w[is_neg] - W_MIN)

            self.W[post, mask] = w