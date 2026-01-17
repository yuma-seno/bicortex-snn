import numpy as np

class BiCortexEngine:
    def __init__(self, n_input: int, n_motor: int, n_mem: int, dt: float = 1.0):
        """
        Bi-Cortex SNN Core Engine
        
        Args:
            n_input (int): 思考野(概念/入力)のニューロン数
            n_motor (int): 思考野(運動/出力)のニューロン数
            n_mem (int): 記憶野(Memory Cortex)のニューロン数
            dt (float): シミュレーションの時間刻み (ms)
        """
        self.n_input = n_input   # Concept Neurons (Gate監視対象)
        self.n_motor = n_motor   # Motor Neurons (Action出力)
        self.n_think = n_input + n_motor
        self.n_mem = n_mem
        self.n_total = self.n_think + n_mem
        self.dt = dt

        # --- 1. ID Mapping (領域定義) ---
        # Thinking Cortex: [0 ... n_input-1] | [n_input ... n_think-1]
        self.idx_think = np.arange(0, self.n_think)
        self.idx_concept = np.arange(0, self.n_input)
        self.idx_motor = np.arange(self.n_input, self.n_think)
        
        # Memory Cortex: [n_think ... n_total-1]
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
        """結合トポロジーの初期化"""
        # A. 記憶野の E/I バランス設定 (Dale's Law)
        n_inh = int(self.n_mem * 0.2)
        self.idx_mem_inh = self.idx_mem[-n_inh:] if n_inh > 0 else []
        self.idx_mem_exc = np.setdiff1d(self.idx_mem, self.idx_mem_inh)

        # B. 結合の初期化
        rng = np.random.default_rng(42)

        # 1. Thinking(Concept) -> Memory (Fixed, Context Injection)
        w_t2m = rng.random((self.n_mem, self.n_input)) * 0.5
        mask_t2m = rng.random((self.n_mem, self.n_input)) < 0.2
        self.W[np.ix_(self.idx_mem, self.idx_concept)] = w_t2m * mask_t2m

        # 2. Memory -> Memory (Plastic, Recurrent LSM)
        w_rec = rng.random((self.n_mem, self.n_mem)) * 0.5
        # 【修正】自己結合(対角成分)を削除して安定化させる
        np.fill_diagonal(w_rec, 0.0) 
        
        mask_rec = rng.random((self.n_mem, self.n_mem)) < 0.1
        # マスクにも対角成分削除を適用
        np.fill_diagonal(mask_rec, False)
        
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec * mask_rec
        
        # 学習許可
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

        # 3. Memory -> Thinking(Motor) (Plastic, Interface)
        self.mask_plastic[np.ix_(self.idx_motor, self.idx_mem)] = True

        # C. 抑制性ニューロンの重みを負にする
        if len(self.idx_mem_inh) > 0:
            self.W[:, self.idx_mem_inh] *= -1.0
            self.W[:, self.idx_mem_inh] *= 2.0 

    def step(self, input_current: np.ndarray, learning: bool = True):
        """1ステップ計算"""
        if input_current.shape[0] != self.n_total:
            raise ValueError(f"Input mismatch: {input_current.shape[0]} != {self.n_total}")

        synaptic_input = self.W @ self.x_fast
        
        is_refractory = self.refractory_count > 0
        self.v = self.v * self.alpha + input_current + synaptic_input
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        spikes = (self.v >= self.v_th).astype(float)
        
        fired_indices = np.where(spikes > 0)[0]
        self.v[fired_indices] = 0.0
        self.refractory_count[fired_indices] = self.refractory_steps

        # Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes

        # Learning
        if learning:
            self._update_weights(spikes)

        return spikes

    def _update_weights(self, spikes):
        """Phase 1.3 で実装"""
        pass