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

* **$x_{fast}$ (即時トレース):** $\tau \approx 5ms$。シナプス後電流(PSC)として使用。
* **$e_{slow}$ (適格性トレース):** $\tau \approx 2000ms$。学習時の因果関係特定に使用。

## 3. Learning Algorithm: 3-Factor Rule (SRG + RL)

本モデルは、従来の「共鳴 (Hebbian)」に加え、「報酬 (Reinforcement)」を取り入れた3要素学習則を採用しています。

### 更新則
$$\Delta w_{ij}(t) = \Delta w_{Hebb}(t) + \Delta w_{RL}(t)$$

#### A. 共鳴学習 (SRG: Semantic Resonance Gating)
思考野が活性化している（ゲートが開いている）時のみ、因果関係を強化する。
$$\Delta w_{Hebb} = \eta \cdot G(t) \cdot Post_i(t) \cdot Pre_{slow, j}(t)$$

#### B. 強化学習 (RL: Dopamine Modulation)
環境からの報酬信号 $R(t)$ に基づき、シナプスを強化または抑制する。
$$\Delta w_{RL} = \eta \cdot R(t) \cdot Pre_{slow, j}(t) \cdot \lambda_{RL}$$

* $R(t) > 0$: **LTP (Long-Term Potentiation)** - 正の報酬による強化。
* $R(t) < 0$: **LTD (Long-Term Depression)** - 罰（誤答）による抑制。
* $R(t) = 0$: 変化なし。

この組み合わせにより、**「意味のある瞬間に（SRG）」**かつ**「結果が良かった行動（RL）」**のみを学習することが可能となる。