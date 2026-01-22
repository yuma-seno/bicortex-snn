# Mathematical Model & Implementation Details

## 1. Neuron Model (LIF with Trace)

### 数式定義
$$v_i(t) = v_i(t-1) \cdot \alpha + I_{ext}(t) + \sum_{j} w_{ij} x_{fast, j}(t-1)$$
$$v_{th} = v_{base}$$

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値(例) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 | 初期値 0.0 |
| $\alpha$ | `self.alpha` | 電圧減衰係数 | $\tau_m=20ms$ |
| $v_{base}$ | `self.v_base` | 基準閾値 | 1.0 ~ 2.0 |

---

## 2. Dual Traces

* **$x_{fast}$:** シナプス後電流(PSC)。信号伝達用。
* **$e_{slow}$:** 適格性トレース。短期的な因果関係の特定に使用される。

## 3. Association Algorithm: Trace-based Linking

記憶野内部の結合変更は、能力の獲得（Learning）ではなく、**一時的な連合の形成（Association Formation）**として定義される。

### 更新則
$$\Delta w_{ij}(t) = \Delta w_{Link}(t) - w_{decay}(t)$$

#### A. 連合形成 (Linking / RL-like)
思考野からの報酬信号 $R(t)$ に基づき、因果関係にあるシナプスを一時的に強化する。
$$\Delta w_{Link} = \eta \cdot R(t) \cdot Pre_{slow, j}(t)$$

* $R(t)$: 思考野が判断した価値（報酬）。
* $Pre_{slow, j}(t)$: 記憶野に残る過去の痕跡。

#### B. 自然忘却 (Decay)
$$w_{new} = w_{old} \cdot (1 - \lambda_{decay})$$

* これにより、記憶野の結合は永続的な知識にはならず、状況が変われば消滅する「短期記憶」としての性質を持つ。

### 動作フロー
1.  **Input:** Interface経由で刺激が入力され、MCに痕跡が残る。
2.  **Reasoning & Action:** TCが推論し、反射または記憶に基づいた行動を行う。
3.  **Association:** TCが報酬を検知すると、MC内の「痕跡」と「現在の状態」が結びつく。
4.  **Decay:** 時間経過とともに結合は解消され、メモリは解放される。