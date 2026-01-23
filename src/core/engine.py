import numpy as np

class BiCortexEngine:
    """
    Bi-Cortex SNN Engine
    
    単一のニューロン基盤（Single Substrate）上に、思考野(Thinking Cortex)と
    記憶野(Memory Cortex)を機能的に配置したSNNエンジン。
    
    [主な機能]
    1. LIFモデルによるスパイク生成
    2. Dual Trace (即時/適格性) による時間的因果の保持
    3. 生理学的な神経調節（Neuromodulation）による自律的な学習
    """

    def __init__(self, 
                 n_input: int, 
                 n_hidden: int, 
                 n_motor: int, 
                 n_mem: int, 
                 dt: float = 1.0, 
                 learning_rate: float = 0.05,
                 gate_ratio: float = 0.02,
                 global_decay: float = 0.001,
                 w_scale_fixed: float = 0.0,
                 w_scale_rec: float = 0.05,
                 seed: int = 42):
        """
        Args:
            n_input (int): 思考野・感覚入力ニューロン数 (Sensory)
            n_hidden (int): 思考野・中間ニューロン数 (Internal/Evaluator)
                ※ 先頭の5つは「Modulator (ドーパミン作動性) ニューロン」として予約されます。
            n_motor (int): 思考野・運動ニューロン数 (Motor)
            n_mem (int): 記憶野・連合ニューロン数 (Associative Memory)
            
            dt (float): シミュレーションの時間刻み (ms)
            learning_rate (float): シナプス可塑性の学習率
            gate_ratio (float): SRGゲートを開くための活動閾値の割合 (0.02 = 2%)
            global_decay (float): 自然忘却率。報酬がない結合を減衰させる係数
            
            w_scale_fixed (float): 思考野内部(固定)の初期結合強度スケール
            w_scale_rec (float): 記憶野内部(可塑)の初期結合強度スケール
            seed (int): 乱数シード
        """
        
        self.rng = np.random.default_rng(seed)
        
        # --- 1. 領域定義 (Parcellation) ---
        # 物理的には1つの配列だが、インデックスで役割を分ける
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_motor = n_motor
        self.n_mem = n_mem
        
        # Thinking Cortex (TC): 固定的な知能領域
        self.n_think = n_input + n_hidden + n_motor
        
        # 全ニューロン数
        self.n_total = self.n_think + n_mem
        
        # --- 2. 神経調節系 (Neuromodulator System) ---
        # Hidden層の先頭5つを「可塑性制御（報酬）ニューロン」として定義
        # これらが発火すると脳全体に学習信号（ドーパミン）が送られる
        self.n_modulator = 5
        if n_hidden < self.n_modulator:
            raise ValueError(f"n_hidden must be at least {self.n_modulator} for modulator neurons.")
        
        # ID: [Input ... | Modulator ... | General Hidden ... | Motor ... | Memory ...]
        self.idx_input = np.arange(0, n_input)
        self.idx_modulator = np.arange(n_input, n_input + self.n_modulator)
        self.idx_hidden_gen = np.arange(n_input + self.n_modulator, n_input + n_hidden)
        self.idx_motor = np.arange(n_input + n_hidden, self.n_think)
        self.idx_mem = np.arange(self.n_think, self.n_total)
        
        # --- 3. ハイパーパラメータ ---
        self.dt = dt
        self.learning_rate = learning_rate
        self.gate_threshold = max(0.1, self.n_input * gate_ratio)
        self.global_decay = global_decay
        
        self.w_scale_fixed = w_scale_fixed
        self.w_scale_rec = w_scale_rec
        
        # SRG状態変数
        self.activity_ma = 0.0
        self.ma_alpha = 0.2
        self.last_gate_status = 0.0

        # --- 4. ニューロン状態変数 (LIF Model) ---
        # v: 膜電位
        self.v = np.zeros(self.n_total)
        self.v_base = 5.0 # 高めの閾値設定（ノイズ耐性）
        
        # 時定数設定
        self.tau_m = 20.0
        self.alpha = np.exp(-dt / self.tau_m)
        self.refractory_count = np.zeros(self.n_total)
        self.refractory_steps = max(1, int(2.0 / dt))

        # --- 5. トレース変数 (Dual Traces) ---
        # x_fast: シナプス後電流 (PSC) -> 情報伝達用 (tau ~ 5ms)
        self.x_fast = np.zeros(self.n_total)
        self.tau_fast = 5.0
        self.decay_fast = np.exp(-dt / self.tau_fast)

        # e_slow: 適格性トレース (Eligibility Trace) -> 学習用 (tau ~ 2000ms)
        self.e_slow = np.zeros(self.n_total)
        self.tau_slow = 2000.0
        self.decay_slow = np.exp(-dt / self.tau_slow)

        # --- 6. 結合行列 (Synapses) ---
        # 全ニューロン間の結合重み
        self.W = np.zeros((self.n_total, self.n_total))
        # 可塑性マスク: Trueの箇所のみ update_weights で更新される
        self.mask_plastic = np.zeros((self.n_total, self.n_total), dtype=bool)
        
        # 初期トポロジー構築
        self._build_topology()

    def _build_topology(self):
        """
        初期の配線を行う。
        - 思考野内部: 固定結合（ランダム初期化、後で上書き可能）
        - 記憶野内部: 可塑的結合（初期値はランダムまたはゼロ）
        """
        # A. Thinking Internal (Fixed)
        if self.n_hidden > 0 and self.w_scale_fixed > 0:
            idx_all_hidden = np.concatenate([self.idx_modulator, self.idx_hidden_gen])
            # Input -> Hidden
            self.W[np.ix_(idx_all_hidden, self.idx_input)] = \
                self.rng.random((self.n_hidden, self.n_input)) * self.w_scale_fixed
            # Hidden -> Motor
            self.W[np.ix_(self.idx_motor, idx_all_hidden)] = \
                self.rng.random((self.n_motor, self.n_hidden)) * self.w_scale_fixed

        # B. Memory Internal (Plastic)
        # MC <-> MC (Recurrent)
        if self.w_scale_rec > 0:
            w_rec = self.rng.random((self.n_mem, self.n_mem)) * self.w_scale_rec
            np.fill_diagonal(w_rec, 0.0) # 自己結合なし
            self.W[np.ix_(self.idx_mem, self.idx_mem)] = w_rec
            self.mask_plastic[np.ix_(self.idx_mem, self.idx_mem)] = True

    def reset_state(self):
        """
        トライアル間の状態リセット。
        短期記憶（膜電位、トレース）をクリアする。
        長期記憶（重みW）は保持される。
        """
        self.v[:] = 0
        self.x_fast[:] = 0
        self.e_slow[:] = 0
        self.refractory_count[:] = 0

    def step(self, input_current: np.ndarray):
        """
        シミュレーションの1ステップを実行する。
        学習（重み更新）も内部状態に基づいて自律的に行われる。
        
        Args:
            input_current (np.ndarray): 外部からの入力電流 [n_total]
        """
        # --- 1. LIF Dynamics (積分発火) ---
        # シナプス入力の計算 (W * x_fast)
        synaptic_input = self.W @ self.x_fast
        
        is_refractory = self.refractory_count > 0
        
        # 膜電位更新: v(t) = v(t-1)*alpha + I_ext + I_syn
        self.v = self.v * self.alpha + input_current + synaptic_input
        self.v[is_refractory] = 0.0
        self.refractory_count = np.maximum(0, self.refractory_count - 1)

        # 発火判定
        spikes = (self.v >= self.v_base).astype(float)
        fired = np.where(spikes > 0)[0]
        
        # リセット
        self.v[fired] = 0.0
        self.refractory_count[fired] = self.refractory_steps
        
        # --- 2. Trace Update (記憶痕跡の更新) ---
        # 即時トレース（通信用）
        self.x_fast = self.x_fast * self.decay_fast + spikes
        # 適格性トレース（学習用）: 長時間残る
        self.e_slow = self.e_slow * self.decay_slow + spikes
        self.e_slow = np.clip(self.e_slow, 0.0, 5.0)

        # --- 3. Semantic Resonance Gating (SRG) ---
        # 思考野(Input)の活動レベルを監視し、ゲート状態を更新
        concept_activity = np.sum(spikes[self.idx_input])
        self.activity_ma = self.activity_ma * (1 - self.ma_alpha) + concept_activity * self.ma_alpha
        self.last_gate_status = 1.0 if self.activity_ma >= self.gate_threshold else 0.0

        # --- 4. Intrinsic Plasticity (自律学習) ---
        # 内部状態に基づいて重みを更新
        self._update_weights_intrinsic(spikes)

        return spikes

    def _update_weights_intrinsic(self, spikes: np.ndarray):
        """
        脳内部の活動のみに基づく学習則。
        外部教師信号は一切使用せず、Modulatorニューロンの活動を報酬として利用する。
        """
        # A. Decay (自然忘却 / Extinction)
        # 報酬がない結合は時間とともに減衰する
        if self.global_decay > 0:
             self.W[self.mask_plastic] *= (1.0 - self.global_decay)

        # B. Neuromodulator Release (ドーパミン放出)
        # 定義されたModulatorニューロンの発火量を計測
        dopamine_level = np.sum(spikes[self.idx_modulator]) * 5.0
        
        # 学習条件: (ゲートが開いている OR ドーパミンが出ている)
        # ここではパブロフ学習のため、ドーパミンが出ていない時は学習をスキップ
        if dopamine_level <= 0: return

        # C. 3-Factor Update Rule
        # Delta W = LearningRate * Dopamine * Pre_Trace
        fired = np.where(spikes > 0)[0]
        pre_trace = self.e_slow
        W_MAX = 5.0; W_MIN = -5.0
        
        # 更新対象: 発火したニューロン(Post)
        targets = fired
        # ドーパミンが強い場合は、発火していなくてもTraceがあるシナプスも強化（遡及的学習）
        if dopamine_level > 1.0:
            active_trace_indices = np.where(self.e_slow > 0.1)[0]
            targets = np.unique(np.concatenate([fired, active_trace_indices]))

        for post in targets:
            mask = self.mask_plastic[post, :]
            if not np.any(mask): continue
            
            w = self.W[post, mask]
            pre = pre_trace[mask]
            
            # 更新量の計算
            delta = self.learning_rate * dopamine_level * pre
            
            # 適用とクリッピング
            new_w = w + delta
            self.W[post, mask] = np.clip(new_w, W_MIN, W_MAX)