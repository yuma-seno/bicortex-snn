# Mathematical Model & Implementation Details

## 1. Neuron Model (LIF with Adaptation & Trace)

### 数式定義
計算効率と安定性を両立させるため、**順応 (Adaptation)** 付きの Leaky Integrate-and-Fire (LIF) モデルを採用する。順応項により、継続的な発火に対して閾値が上昇し、無限ループ（てんかん発作的活動）を自然に鎮火させる。

**膜電位更新:**
$$v_i(t) = v_i(t-1) \cdot \alpha_{decay} + I_{ext}(t) + I_{syn}(t)$$
$$I_{syn}(t) = \sum_{j} w_{ij} x_{fast, j}(t-1)$$

**順応変数更新:**
$$A_i(t) = A_i(t-1) \cdot \alpha_{adapt} + S_i(t-1) \cdot \delta_{adapt}$$

* **発火条件:**
  $$v_i(t) \ge v_{base} + A_i(t)$$
  * 条件を満たす場合、$S_i(t) = 1, v_i(t) = 0$ とし、その後不応期に入る。

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値 (Typical) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 | Initial 0.0 |
| $\alpha_{decay}$ | `self.alpha` | 電圧減衰係数 | $\tau_m=20ms$ |
| $v_{base}$ | `self.v_base` | 基礎閾値 | **5.0** |
| $A_i(t)$ | `self.adaptation` | 順応（疲労）変数 | - |
| $\delta_{adapt}$ | `self.adaptation_step` | 発火ごとの閾値上昇量 | **0.3** (Balance) |
| $\alpha_{adapt}$ | `self.decay_adapt` | 順応の回復係数 | $\tau_{adapt}=100ms$ |
| $t_{ref}$ | `self.refractory_steps`| 不応期 | **2.0ms** |

---

## 2. Dual Traces (2つのトレース変数)

時間的な因果関係を学習するため、時定数の異なる2つのトレース変数を使用する。

### 2.1 即時トレース ($x_{fast}$)
シナプス後電流 (PSC) の立ち上がりを表現する速いトレース。
$$x_{fast, i}(t) = x_{fast, i}(t-1) \cdot \alpha_{fast} + S_i(t)$$

* **時定数:** $\tau_{fast} = 5.0ms$
* **役割:** ニューロン間の信号伝達。

### 2.2 適格性トレース ($e_{slow}$)
学習用の長期記憶痕跡。報酬やゲート信号が遅れて到達した際に、過去の原因を特定するために用いられる。
$$e_{slow, i}(t) = e_{slow, i}(t-1) \cdot \alpha_{slow} + S_i(t)$$

* **時定数:** $\tau_{slow} = 2000.0ms$ (2.0s)
* **役割:** 時間差学習 (Temporal Credit Assignment)。

---

## 3. Learning Algorithm: Semantic Resonance Gating (SRG)

### 3.1 ゲート信号 $G(t)$ (Moving Average Logic)
思考野の**概念ニューロン群 ($Concept$)** の活動総量を監視し、その**移動平均 (Moving Average)** が閾値を超えた場合にのみ学習を許可する。

$$A_{ma}(t) = A_{ma}(t-1) \cdot (1 - \alpha_{ma}) + \left( \sum_{k \in TC_{concept}} S_k(t) \right) \cdot \alpha_{ma}$$

$$G(t) = 1 \quad \text{if} \quad A_{ma}(t) \ge \theta_{gate} \quad \text{else} \quad 0$$

* **平滑化係数:** $\alpha_{ma} = 0.2$
* **ゲート閾値:** $\theta_{gate} = \max(0.1, N_{concept} \times \text{gate\_ratio})$
    * **変更点:** 最小値を `1.0` から `0.1` へ変更。不応期により発火率が制限される小規模ネットワークでも学習可能にした。

### 3.2 リカレント結合の更新則
ゲートが開いている時に、記憶野内部の結合を更新する。重み減衰（忘却）とクリッピングを含む。

**減衰 (Global Decay):**
$$w_{mn}(t)' = w_{mn}(t-1) \cdot (1 - \lambda_{decay})$$

**強化 (Hebbian Learning with Gating):**
$$\Delta w_{mn}(t) = \eta \cdot G(t) \cdot S_{mem, m}(t) \cdot e_{slow, n}(t)$$

**統合とクリッピング:**
$$w_{mn}(t) = \text{clip}\left( w_{mn}(t)' + \Delta w_{mn}(t), \ -w_{max}, \ w_{max} \right)$$

* $m \in MC$ (Postsynaptic / 現在の記憶活動)
* $n \in MC$ (Presynaptic / 過去の記憶痕跡)
* $\eta$: 学習率 (`learning_rate`). Traceの蓄積を考慮し `0.001`~`0.005` 程度に設定。
* $\lambda_{decay}$: 重み減衰率 (`global_decay`).
* $w_{max}$: 重み最大値 (`w_max_clip`). **1.0** 程度に制限し、過剰学習を防ぐ。