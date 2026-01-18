# Mathematical Model & Implementation Details

## 1. Neuron Model (Adaptive LIF)

計算コストを最小化するため、離散時間における積分発火モデルに、**適応的閾値 (Adaptive Threshold)** を導入しています。

### 数式定義
$$v_i(t) = v_i(t-1) \cdot \alpha + I_{ext}(t) + \sum_{j} w_{ij} x_{fast, j}(t-1)$$
$$v_{th}(t) = v_{base} + v_{adaptive}(t)$$
$$v_{adaptive}(t) = v_{adaptive}(t-1) \cdot \alpha_{adapt} + \beta \cdot S_i(t-1)$$

発火条件: $v_i(t) \ge v_{th}(t)$

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値(例) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 | 初期値 0.0 |
| $\alpha$ | `self.alpha` | 電圧減衰係数 | $\tau_m=20ms$ |
| $v_{base}$ | `self.v_base` | 基準閾値 | 1.0 |
| $v_{adaptive}$ | `self.v_th_adaptive` | 閾値オフセット | 発火ごとに上昇 |
| $\beta$ | `self.adaptation_beta` | 閾値上昇率（疲れやすさ） | 0.5 |
| $\alpha_{adapt}$ | `self.adaptation_decay` | 疲労回復係数 | $\tau_{adapt}=50ms$ |

---

## 2. Dual Traces (2つのトレース変数)

* **$x_{fast}$ (即時トレース):** $\tau \approx 5ms$。シナプス後電流(PSC)として使用。キレの良い応答のため短めに設定。
* **$e_{slow}$ (適格性トレース):** $\tau \approx 2000ms$。学習時の因果関係特定に使用。

---

## 3. Learning Algorithm: SRG (Soft-bound Hebbian)

（変更なし）
ただし、学習対象（Plastic Mask）はインターフェース結合（Memory -> Motor）のみに限定される。

---
（以下変更なし）