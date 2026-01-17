import numpy as np

class BiCortexEngine:
    def __init__(self, n_think: int, n_mem: int, dt: float = 1.0):
        """
        Bi-Cortex SNN Core Engine
        
        Args:
            n_think (int): 思考野(Thinking Cortex)のニューロン数
            n_mem (int): 記憶野(Memory Cortex)のニューロン数
            dt (float): シミュレーションの時間刻み (ms)
        """
        self.n_think = n_think
        self.n_mem = n_mem
        self.n_total = n_think + n_mem
        self.dt = dt

        # --- ID Mapping ---
        self.idx_think = np.arange(0, n_think)
        self.idx_mem = np.arange(n_think, self.n_total)

        # --- Neuron Parameters (LIF) ---
        self.v = np.zeros(self.n_total)          # 膜電位
        self.v_th = 1.0                          # 発火閾値
        self.tau_m = 20.0                        # 膜時定数 (ms)
        self.alpha = np.exp(-dt / self.tau_m)    # 減衰係数
        
        # 不応期 (Refractory Period)
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = int(2.0 / dt)    # 2ms の不応期

        # --- Dual Traces ---
        # 1. Immediate Trace (x_fast): シナプス伝達用 (~20ms)
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 20.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        # 2. Eligibility Trace (e_slow): 因果関係学習用 (~2000ms)
        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0  # 2秒
        self.decay_slow = np.exp(-dt / self.tau_slow)

        # --- Weights ---
        # W[post, pre] の形式 (行: post, 列: pre)
        self.W = np.zeros((self.n_total, self.n_total))

    def step(self, input_current: np.ndarray, learning: bool = True):
        """
        1ステップのダイナミクス計算
        
        Args:
            input_current (np.ndarray): 外部入力電流 (Shape: [n_total])
            learning (bool): 学習(SRG)を行うかどうか
        
        Returns:
            spikes (np.ndarray): 発火したニューロンのマスク (0 or 1)
        """
        if input_current.shape[0] != self.n_total:
            raise ValueError(f"Input shape mismatch: expected {self.n_total}, got {input_current.shape[0]}")

        # 1. 膜電位更新 (LIF)
        # v(t) = v(t-1) * alpha + Input + (W @ x_fast)
        # ※ x_fast は前ステップのスパイクの影響を平滑化したもの(PSC: Post Synaptic Current)として扱う
        synaptic_input = self.W @ self.x_fast
        
        # 不応期中のニューロンは電位更新しない（または0に固定）
        is_refractory = self.refractory_count > 0
        
        # 更新
        self.v = self.v * self.alpha + input_current + synaptic_input
        
        # 不応期リセット & カウントダウン
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        # 2. 発火判定
        spikes = (self.v >= self.v_th).astype(float)
        
        # 発火したニューロンの処理
        fired_indices = np.where(spikes > 0)[0]
        self.v[fired_indices] = 0.0                # リセット
        self.refractory_count[fired_indices] = self.refractory_steps # 不応期セット

        # 3. トレース更新
        self.x_fast = self.x_fast * self.decay_fast + spikes
        self.e_slow = self.e_slow * self.decay_slow + spikes

        # 4. 学習 (SRG) - Phase 1.3 で実装予定
        if learning:
            self._update_weights(spikes)

        return spikes

    def _update_weights(self, spikes):
        """学習則 (Phase 1.3 で実装)"""
        pass