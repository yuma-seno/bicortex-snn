# Mathematical Model & Implementation Details

## 1. Neuron Model (LIF: Leaky Integrate-and-Fire)

計算コストを最小化するため、離散時間における積分発火モデルを採用しています。

### 数式定義
$$v_i(t) = v_i(t-1) \cdot \alpha + I_{ext}(t) + \sum_{j} w_{ij} x_{fast, j}(t-1)$$

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値(例) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 (Membrane Potential) | 初期値 0.0 |
| $\alpha$ | `self.alpha` | 減衰係数 ($\exp(-dt/\tau_m)$) | $\tau_m=20ms$ |
| $I_{ext}(t)$ | `input_current` | 外部入力電流 | - |
| $v_{th}$ | `self.v_th` | 発火閾値 | 1.0 |
| $Refractory$| `refractory_steps` | 不応期ステップ数 | $\max(1, 2.0ms/dt)$ |

---

## 2. Dual Traces (2つのトレース変数)

時間的信用割当問題（Temporal Credit Assignment）を解決するため、時定数の異なる2つのトレース変数を保持します。

* **$x_{fast}$ (即時トレース):** $\tau \approx 20ms$。シナプス後電流(PSC)として使用。
* **$e_{slow}$ (適格性トレース):** $\tau \approx 2000ms$。学習時の因果関係特定に使用。

---

## 3. Learning Algorithm: SRG (Soft-bound Hebbian)

Phase 1.3 (v1.2) より、単純加算ではなく、重みの飽和を防ぐ **Soft-bound** 方式と **Global Decay** を採用しました。

### 3.1 ゲート信号 $G(t)$ (Moving Average & Ratio)
思考野の活動の移動平均（MA）が、全思考ニューロン数に対する所定の割合（Ratio）を超えた場合にゲートが開きます。

$$Activity_{MA}(t) = Activity_{MA}(t-1) \cdot (1-\alpha_{ma}) + \sum S_{think}(t) \cdot \alpha_{ma}$$
$$G(t) = 1 \quad \text{if} \quad Activity_{MA}(t) \ge (N_{think} \cdot \text{gate\_ratio}) \quad \text{else} \quad 0$$

### 3.2 重み更新則 (Soft-bound)
重みが上下限（$W_{max}, W_{min}$）に近づくにつれて更新量を減衰させ、自然な境界維持を行います。

**A. 興奮性シナプス ($Pre \in Exc$) の場合:**
$$\Delta w_{ij} = (W_{max} - w_{ij}) \cdot \eta \cdot G(t) \cdot e_{slow, j}(t)$$

**B. 抑制性シナプス ($Pre \in Inh$) の場合:**
$$\Delta w_{ij} = -1 \cdot (w_{ij} - W_{min}) \cdot \eta \cdot G(t) \cdot e_{slow, j}(t)$$
※ 抑制性は「より負の方向」へ強化される。

### 3.3 全体減衰 (Global Weight Decay)
可塑的結合（Plastic）に対して、毎ステップわずかな減衰を適用し、恒常性を維持します。
$$w_{ij}(t) = w_{ij}(t) \cdot (1 - \text{global\_decay})$$

---

## 4. Known Limitations & Future Consideration

現在の数学モデルにおける既知の制約事項です（Phase 1では許容）。

1.  **STDPのタイミング依存性:**
    * 現在は Pre(Slow) $\times$ Post(Spike) の因果のみを見ており、ミリ秒単位のタイミング差（Postが先かPreが先か）による厳密なSTDPは実装していない。数秒単位の行動学習には現状で十分と判断している。
2.  **数値安定性:**
    * $\tau_{slow}=2000ms$ の減衰率は非常に 1.0 に近いため、浮動小数点の丸め誤差の影響を受ける可能性がある。
3.  **学習ループの計算効率:**
    * 現在は可読性を優先し Python ループ内でマスク処理を行っている。大規模化の際はベクトル化が必要。