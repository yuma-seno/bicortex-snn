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
                 global_decay: float = 0.01,
                 w_scale_fixed: float = 0.0,
                 w_scale_rec: float = 0.05,
                 adaptation_beta: float = 0.5,
                 adaptation_tau: float = 50.0,
                 seed: int = 42):
        """
        Bi-Cortex SNN Core Engine v2.4 (3-Factor Learning Rule)
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
        # A. Thinking Internal (Fixed)
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
        if self.w_scale_rec > 0:
            w_rec = self.rng.random((self.n_mem, self.n_mem)) * self.w_scale_rec
            np.fill_diagonal(w_rec, 0.0)
            self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec
            self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # D. Interface (Memory -> Motor) [FIXED by default, but customizable]
        # self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = False

        # E. Dale's Law (Inhibition)
        n_inh = int(self.n_mem * 0.2)
        if n_inh > 0:
            self.idx_mem_inh = self.idx_mem[-n_inh:]
            self.W[:, self.idx_mem_inh] *= -1.0
            self.W[:, self.idx_mem_inh] *= 2.0
        else:
            self.idx_mem_inh = np.array([], dtype=int)

    def step(self, input_current: np.ndarray, reward: float = 0.0, learning: bool = True):
        """
        Step function with 3-Factor Learning Rule (Dopamine Modulation)
        :param reward: >0 (LTP/Reward), <0 (LTD/Punishment), 0 (No modulation)
        """
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
        
        self.v_th_adaptive[fired] += self.adaptation_beta
        self.v_th_adaptive *= self.adaptation_decay

        # 3. Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes

        # 4. Learning (SRG + RL)
        if learning:
            self._update_weights(spikes, reward)
            
        if self.global_decay > 0:
            self.W[self.mask_plastic] *= (1.0 - self.global_decay)

        return spikes

    def _update_weights(self, spikes, reward):
        # --- A. Gate Control (SRG) ---
        concept_activity = np.sum(spikes[self.idx_input])
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        
        G_t = 1.0 if self.activity_ma >= self.gate_threshold else 0.0
        self.last_gate_status = G_t

        if G_t == 0.0 and reward == 0.0: return # ゲートも閉じて報酬もなければ何もしない

        fired = np.where(spikes > 0)[0]
        
        # --- B. Learning Rule ---
        pre_trace = self.e_slow
        W_MAX = 2.0; W_MIN = -2.0
        
        # 学習対象: 発火したニューロンへの入力 (Post-synaptic driven)
        # または、報酬がある場合は全可塑性シナプスを対象にすることも考えられるが、
        # ここでは計算効率のため「発火したニューロン」または「強い痕跡があるニューロン」を中心に更新する
        
        targets = fired
        if reward != 0.0:
            # 報酬があるときは、発火していなくても「痕跡(Eligibility Trace)」が残っているニューロンも更新対象にする
            # (数秒前の行動を強化するため)
            active_trace_indices = np.where(self.e_slow > 0.1)[0]
            targets = np.unique(np.concatenate([fired, active_trace_indices]))

        for post in targets:
            mask = self.mask_plastic[post, :]
            if not np.any(mask): continue
            
            w = self.W[post, mask]
            pre = pre_trace[mask]
            
            # Delta計算:
            # 1. Hebbian (SRG): ゲートが開いているとき、共起したら強化
            delta_hebb = self.learning_rate * G_t * pre
            
            # 2. Reinforcement (RL): 報酬があるとき、痕跡に比例して強化/抑制
            # reward正ならプラス、負ならマイナス
            delta_rl = self.learning_rate * reward * pre * 5.0 # RLの影響を強めに
            
            delta = delta_hebb + delta_rl

            # Soft-bound Update
            is_pos = w >= 0
            if np.any(is_pos):
                w[is_pos] += delta[is_pos] * (W_MAX - w[is_pos])
            is_neg = w < 0
            if np.any(is_neg):
                w[is_neg] -= delta[is_neg] * (w[is_neg] - W_MIN)

            self.W[post, mask] = w