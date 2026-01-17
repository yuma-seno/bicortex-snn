import numpy as np

class BiCortexEngine:
    def __init__(self, n_input: int, n_motor: int, n_mem: int, dt: float = 1.0, 
                 learning_rate: float = 0.01, gate_threshold: float = 3.0):
        """
        Bi-Cortex SNN Core Engine
        
        Args:
            n_input (int): 思考野(概念/入力)のニューロン数
            n_motor (int): 思考野(運動/出力)のニューロン数
            n_mem (int): 記憶野(Memory Cortex)のニューロン数
            dt (float): シミュレーションの時間刻み (ms)
            learning_rate (float): 学習率 (eta)
            gate_threshold (float): ゲートを開くための概念ニューロン最小発火数
        """
        self.n_input = n_input
        self.n_motor = n_motor
        self.n_think = n_input + n_motor
        self.n_mem = n_mem
        self.n_total = self.n_think + n_mem
        self.dt = dt
        
        # --- Hyperparameters ---
        self.learning_rate = learning_rate
        self.gate_threshold = gate_threshold

        # --- 1. ID Mapping ---
        self.idx_think = np.arange(0, self.n_think)
        self.idx_concept = np.arange(0, self.n_input)
        self.idx_motor = np.arange(self.n_input, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)

        # --- 2. Neuron Parameters (LIF) ---
        self.v = np.zeros(self.n_total)
        self.v_th = 1.0
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = int(2.0 / dt)

        # --- 3. Dual Traces ---
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 20.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0
        self.decay_slow = np.exp(-dt / self.tau_slow)

        # --- 4. Topology & Weights ---
        self.W = np.zeros((self.n_total, self.n_total))
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)
        
        self._build_topology()

    def _build_topology(self):
        # ... (Phase 1.2 と同じ内容) ...
        n_inh = int(self.n_mem * 0.2)
        self.idx_mem_inh = self.idx_mem[-n_inh:] if n_inh > 0 else []
        self.idx_mem_exc = np.setdiff1d(self.idx_mem, self.idx_mem_inh)

        rng = np.random.default_rng(42)

        # 1. Thinking(Concept) -> Memory
        w_t2m = rng.random((self.n_mem, self.n_input)) * 0.5
        mask_t2m = rng.random((self.n_mem, self.n_input)) < 0.2
        self.W[np.ix_(self.idx_mem, self.idx_concept)] = w_t2m * mask_t2m

        # 2. Memory -> Memory
        w_rec = rng.random((self.n_mem, self.n_mem)) * 0.5
        np.fill_diagonal(w_rec, 0.0) 
        mask_rec = rng.random((self.n_mem, self.n_mem)) < 0.1
        np.fill_diagonal(mask_rec, False)
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec * mask_rec
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # 3. Memory -> Thinking(Motor)
        self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = True

        # C. Inhibition
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

        return spikes

    def _update_weights(self, spikes):
        """
        Semantic Resonance Gating (SRG) Learning Rule
        Delta W = eta * G(t) * Post(t) * Pre_slow(t)
        """
        # A. ゲート信号 G(t) の計算
        # 概念ニューロン(Concept)の発火数をカウント
        concept_activity = np.sum(spikes[self.idx_concept])
        
        # 閾値を超えたらゲートが開く (1.0), それ以外は閉じる (0.0)
        G_t = 1.0 if concept_activity >= self.gate_threshold else 0.0
        
        if G_t == 0.0:
            return  # ゲートが閉じていれば計算省略 (高速化)

        # B. 重み更新行列の計算 (Delta W)
        # Post(t) [N x 1]  * Pre_slow(t) [1 x N]  =  [N x N]
        # ※ 学習対象は mask_plastic が True の場所のみ
        
        # 発火したニューロン(Post)のみ計算すればよい
        fired_indices = np.where(spikes > 0)[0]
        
        if len(fired_indices) == 0:
            return

        # Pre側は 全ニューロンの slow trace を使う
        pre_trace = self.e_slow
        
        # 行列演算ループを避けるため、発火したPostニューロンに対してのみ更新
        # W[post, :] += eta * G * pre_trace
        for post_idx in fired_indices:
            # 更新対象の結合マスクを取得
            target_mask = self.mask_plastic[post_idx, :]
            
            if np.any(target_mask):
                # Delta W の加算
                self.W[post_idx, target_mask] += \
                    self.learning_rate * G_t * pre_trace[target_mask]

        # C. 重みのクリッピング (発散防止)
        # 正の重みは [0, 1.0], 負の重み(抑制性)は [-2.0, 0] 程度に収めたいが、
        # ここでは単純に絶対値が大きくなりすぎないように制限
        # ※ ただし、抑制性ニューロンが出発点の重みは負のまま維持する必要がある
        
        # 簡易的なクリッピング: 上限のみ設ける (抑制性の強化は許容、または別途制限)
        self.W[self.mask_plastic] = np.clip(self.W[self.mask_plastic], -2.0, 2.0)