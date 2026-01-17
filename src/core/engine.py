import numpy as np

class BiCortexEngine:
    def __init__(self, n_input: int, n_motor: int, n_mem: int, dt: float = 1.0, 
                 learning_rate: float = 0.01, gate_threshold: float = 3.0):
        """
        Bi-Cortex SNN Core Engine (Phase 1 Final)
        
        Args:
            n_input (int): 思考野(概念/入力)のニューロン数
            n_motor (int): 思考野(運動/出力)のニューロン数
            n_mem (int): 記憶野(Memory Cortex)のニューロン数
            dt (float): シミュレーションの時間刻み (ms)
            learning_rate (float): 学習率
            gate_threshold (float): ゲートを開くための活動閾値
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
        """結合トポロジーの構築"""
        # A. 記憶野の E/I バランス設定 (Dale's Law)
        n_inh = int(self.n_mem * 0.2)
        self.idx_mem_inh = self.idx_mem[-n_inh:] if n_inh > 0 else np.array([], dtype=int)
        self.idx_mem_exc = np.setdiff1d(self.idx_mem, self.idx_mem_inh)

        rng = np.random.default_rng(42)

        # 1. Thinking(Concept) -> Memory (Fixed)
        w_t2m = rng.random((self.n_mem, self.n_input)) * 0.5
        mask_t2m = rng.random((self.n_mem, self.n_input)) < 0.2
        self.W[np.ix_(self.idx_mem, self.idx_concept)] = w_t2m * mask_t2m

        # 2. Memory -> Memory (Plastic, Recurrent)
        w_rec = rng.random((self.n_mem, self.n_mem)) * 0.5
        np.fill_diagonal(w_rec, 0.0) # 自己結合削除
        
        mask_rec = rng.random((self.n_mem, self.n_mem)) < 0.1
        np.fill_diagonal(mask_rec, False)
        
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec * mask_rec
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # 3. Memory -> Thinking(Motor) (Plastic)
        self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = True

        # C. 初期重みへの Dale's Law 適用 (抑制性を負にする)
        if len(self.idx_mem_inh) > 0:
            self.W[:, self.idx_mem_inh] *= -1.0
            self.W[:, self.idx_mem_inh] *= 2.0 

    def step(self, input_current: np.ndarray, learning: bool = True):
        """1ステップ計算"""
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
        """
        # A. ゲート信号 G(t)
        concept_activity = np.sum(spikes[self.idx_concept])
        G_t = 1.0 if concept_activity >= self.gate_threshold else 0.0
        
        if G_t == 0.0:
            return

        # B. 重み更新 (Delta W)
        fired_indices = np.where(spikes > 0)[0]
        if len(fired_indices) == 0:
            return

        pre_trace = self.e_slow
        
        for post_idx in fired_indices:
            target_mask = self.mask_plastic[post_idx, :]
            if np.any(target_mask):
                # 単純加算 (Hebbian like)
                self.W[post_idx, target_mask] += \
                    self.learning_rate * G_t * pre_trace[target_mask]

        # C. 制約とクリッピング (Dale's Law Preservation)
        # ここが修正ポイント: 興奮性と抑制性を別々にクリップする
        
        # 1. 全ての重み(絶対値)が爆発しないようにする
        # (まだ符号は気にしない)
        # self.W = np.clip(self.W, -5.0, 5.0) # 安全策

        # 2. 興奮性ニューロン由来の重み (正の値に保つ)
        # 対象: idx_concept(Fixedだけど念のため), idx_motor(出力はない), idx_mem_exc
        # ※ Wの列(Column)がPreニューロン
        
        # 興奮性列のマスク作成
        col_exc = np.concatenate([self.idx_concept, self.idx_motor, self.idx_mem_exc])
        # [0, 2.0] にクリップ
        self.W[:, col_exc] = np.clip(self.W[:, col_exc], 0.0, 2.0)

        # 3. 抑制性ニューロン由来の重み (負の値に保つ)
        if len(self.idx_mem_inh) > 0:
            # [-2.0, 0] にクリップ (0を超えさせない=反転防止)
            self.W[:, self.idx_mem_inh] = np.clip(self.W[:, self.idx_mem_inh], -2.0, 0.0)