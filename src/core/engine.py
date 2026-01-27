import numpy as np

class BiCortexEngine:
    """
    Bi-Cortex SNN Engine (Original Spec Compliant)
    
    Concept:
    - Thinking Cortex: Sensory -> Concept -> Motor (Fixed/Pre-trained)
    - Memory Cortex: Reservoir with E/I Balance (Plastic)
    - Learning: Semantic Resonance Gating (Concept Activity -> Plasticity)
    """

    def __init__(self, 
                 n_sensory: int, 
                 n_concept: int, 
                 n_motor: int, 
                 n_mem: int, 
                 dt: float = 1.0, 
                 learning_rate: float = 0.05,
                 gate_ratio: float = 0.1, # 概念ニューロンの何割が発火したらGateを開くか
                 global_decay: float = 0.001,
                 seed: int = 42):
        
        self.rng = np.random.default_rng(seed)
        
        # --- 1. 領域定義 (Parcellation) ---
        self.n_sensory = n_sensory
        self.n_concept = n_concept
        self.n_motor = n_motor
        self.n_mem = n_mem
        
        # Thinking Cortex (TC)
        self.n_think = n_sensory + n_concept + n_motor
        self.n_total = self.n_think + n_mem
        
        # ID Ranges
        self.idx_sensory = np.arange(0, n_sensory)
        self.idx_concept = np.arange(n_sensory, n_sensory + n_concept)
        self.idx_motor = np.arange(n_sensory + n_concept, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        # Memory E/I Balance (80% Exc, 20% Inh)
        self.n_exc = int(n_mem * 0.8)
        self.n_inh = n_mem - self.n_exc
        # メモリ領域内の相対インデックスで管理
        self.mem_exc_mask = np.zeros(n_mem, dtype=bool)
        self.mem_exc_mask[:self.n_exc] = True
        self.mem_inh_mask = ~self.mem_exc_mask
        
        # --- 2. ハイパーパラメータ ---
        self.dt = dt
        self.learning_rate = learning_rate
        self.gate_threshold = max(1.0, n_concept * gate_ratio)
        self.global_decay = global_decay
        
        # SRG状態変数
        self.activity_ma = 0.0
        self.ma_alpha = 0.2
        self.is_gating = False

        # --- 3. ニューロン状態変数 ---
        self.v = np.zeros(self.n_total)
        self.v_base = 5.0 
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(2.0 / dt))

        # --- 4. Trace変数 ---
        self.e_trace = np.zeros(self.n_total)
        self.tau_trace = 2000.0 
        self.decay_trace = np.exp(-dt / self.tau_trace)

        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 5.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        # --- 5. 結合行列 (Tabula Rasa) ---
        self.W = np.zeros((self.n_total, self.n_total))
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)

    def init_memory_reservoir(self, density=0.1, spectral_radius=0.9):
        """
        記憶野をリザーバとして初期化する (Dale's Law準拠)
        - 興奮性ニューロンからの出力は正
        - 抑制性ニューロンからの出力は負
        """
        # ローカルな重み行列を作成
        W_mem = np.zeros((self.n_mem, self.n_mem))
        
        # ランダム結合
        mask = self.rng.random((self.n_mem, self.n_mem)) < density
        weights = self.rng.random((self.n_mem, self.n_mem))
        
        # Dale's Law適用
        # 列(Pre)が興奮性なら正、抑制性なら負
        W_mem[:, self.mem_exc_mask] = weights[:, self.mem_exc_mask]      # Exc -> Any
        W_mem[:, self.mem_inh_mask] = -weights[:, self.mem_inh_mask]     # Inh -> Any
        
        # スペクトル半径の調整
        W_mem *= mask
        radius = np.max(np.abs(np.linalg.eigvals(W_mem)))
        if radius > 0:
            W_mem *= (spectral_radius / radius)
            
        # グローバル行列へ適用
        self.W[np.ix_(self.idx_mem, self.idx_mem)] = W_mem
        
        # 自己結合は可塑的とする (Short-term context maintenance)
        self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = (W_mem != 0)

    def reset_state(self):
        self.v[:] = 0
        self.x_fast[:] = 0
        self.e_trace[:] = 0
        self.refractory_count[:] = 0

    def step(self, input_current: np.ndarray):
        # 1. LIF Update
        synaptic_input = self.W @ self.x_fast
        self.v = self.v * self.alpha + input_current + synaptic_input
        
        self.v[self.refractory_count > 0] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        spikes = (self.v >= self.v_base).astype(float)
        fired = np.where(spikes > 0)[0]
        
        self.v[fired] = 0.0
        self.refractory_count[fired] = self.refractory_steps
        
        # 2. Trace Update
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_trace = self.e_trace * self.decay_trace + spikes
        
        # 3. Semantic Resonance Gating (SRG)
        # 監視対象: Conceptニューロンのみ
        concept_activity = np.sum(spikes[self.idx_concept])
        
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        
        # Gate判定
        self.is_gating = (self.activity_ma >= self.gate_threshold)

        # 4. Learning
        self._update_weights_srg(spikes)

        return spikes

    def _update_weights_srg(self, spikes: np.ndarray):
        """
        Original SRG Rule:
        Delta W_mj = eta * G(t) * S_motor(t) * e_slow_m(t)
        
        ゲートが開いている時に、「過去の記憶(m)」と「現在の行動/思考(j)」を結びつける。
        """
        # A. Decay (Stabilization)
        if self.global_decay > 0:
             self.W[self.mask_plastic] *= (1.0 - self.global_decay)
        
        if not self.is_gating:
            return

        fired = np.where(spikes > 0)[0]
        
        # 更新対象: 
        # 主に Memory(Pre) -> Motor(Post) のInterface結合
        # および Memory(Pre) -> Memory(Post) のリザーバ内部結合
        
        for post in fired:
            # Plasticな結合を持っているかチェック
            mask_p = self.mask_plastic[post, :]
            if not np.any(mask_p): continue
            
            # Pre-synaptic Trace (Eligibility)
            pre_trace = self.e_trace[mask_p]
            
            # Hebbian Update: G(t) * Post * Pre
            delta = self.learning_rate * pre_trace
            
            w = self.W[post, mask_p]
            new_w = w + delta
            
            # Clip
            self.W[post, mask_p] = np.clip(new_w, -5.0, 5.0)