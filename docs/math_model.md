# Mathematical Model & Implementation Details

## 1. Neuron Model (LIF with Trace)

### 数式定義
計算効率とスパースな発火特性を得るため、Leaky Integrate-and-Fire (LIF) モデルを採用する。

$$v_i(t) = v_i(t-1) \cdot \alpha_{decay} + I_{ext}(t) + I_{syn}(t)$$
$$I_{syn}(t) = \sum_{j} w_{ij} x_{fast, j}(t-1)$$

* **発火条件:** $v_i(t) \ge v_{th}$ の場合、$S_i(t) = 1, v_i(t) = 0$ とし、その後不応期に入る。

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値 (Default) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 | Initial 0.0 |
| $\alpha_{decay}$ | `self.alpha` | 電圧減衰係数 | $\exp(-dt/\tau_m), \tau_m=20ms$ |
| $v_{th}$ | `self.v_base` | 発火閾値 | **5.0** (ノイズ耐性のため高めに設定) |
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
思考野の**概念ニューロン群 ($Concept$)** の活動総量を監視し、その**移動平均 (Moving Average)** が閾値を超えた場合にのみ学習を許可する。これにより、瞬間的なノイズによる誤学習を防ぐ。

$$A_{ma}(t) = A_{ma}(t-1) \cdot (1 - \alpha_{ma}) + \left( \sum_{k \in TC_{concept}} S_k(t) \right) \cdot \alpha_{ma}$$

$$G(t) = 1 \quad \text{if} \quad A_{ma}(t) \ge \theta_{gate} \quad \text{else} \quad 0$$

* **平滑化係数:** $\alpha_{ma} = 0.2$
* **ゲート閾値:** $\theta_{gate} = \max(1.0, N_{concept} \times 0.1)$ (概念ニューロンの10%程度が活性化している状態)

### 3.2 インターフェース結合の更新則
ゲートが開いている時に、「過去の文脈」と「現在の行動」を関連付ける。

$$\Delta w_{mj}(t) = \eta \cdot G(t) \cdot S_{motor, j}(t) \cdot e_{slow, m}(t)$$

* $m \in MC$ (Presynaptic / 記憶野)
* $j \in TC_{motor}$ (Postsynaptic / 思考野運動)
* $\eta$: 学習率 (`self.learning_rate` = 0.05)
* **安定化処理 (Global Weight Decay):**
    * 重みの発散を防ぐため、毎ステップわずかな減衰を適用する。
    * $w_{mj} \leftarrow w_{mj} \cdot (1 - \lambda_{decay})$
    * $\lambda_{decay} = 0.001$