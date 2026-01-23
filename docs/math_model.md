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
| $v_{base}$ | `self.v_base` | 基準閾値 | **5.0** (ノイズ耐性のため高めに設定) |

---

## 2. Dual Traces (2つのトレース変数)

* **$x_{fast}$ (即時トレース):** $\tau \approx 5ms$。シナプス後電流(PSC)。信号伝達を担う。
* **$e_{slow}$ (適格性トレース):** $\tau \approx 2000ms$。学習用。ニューロン発火後、長時間その「痕跡」を残し、遅延した報酬との結びつきを可能にする。

## 3. Learning Algorithm: Semantic Resonance Gating (SRG)

本アーキテクチャ独自の学習制御メカニズム。
「常に学習する」のではなく、**「意味がある（共鳴している）時だけゲートを開き、痕跡を結びつける」** ことで、低コストかつノイズに強い学習を実現する。

### 3.1 ゲート信号と調節信号
SRGは以下の2つの信号によって制御される。

1.  **Thinking Activity Gate ($G(t)$):**
    思考野が活発に活動している（何らかの概念を処理している）かどうか。
    $$G(t) = 1 \quad \text{if} \quad Activity_{TC}(t) \ge Threshold \quad \text{else} \quad 0$$

2.  **Neuromodulator Signal ($D(t)$):**
    思考野の本能回路（Modulatorニューロン）が放出したドーパミン量。
    $$D(t) = \sum S_{modulator}(t)$$

### 3.2 3要素更新則 (3-Factor Rule)
シナプス結合の更新 $\Delta w$ は、ゲートが開いている、または強い調節信号がある状態でのみ発生する。

$$\Delta w_{Link} = \eta \cdot D(t) \cdot Pre_{slow, j}(t)$$

* **共鳴条件:** $G(t)=1$ または $D(t) > 0$ の時のみ計算が実行される。
* **強化:** ドーパミン ($D$) と過去の痕跡 ($Pre$) が共鳴したシナプスが強化される。
* **忘却:** 共鳴がない状態では、シナプスは自然減衰 ($Decay$) する。

### 動作フロー
1.  **Trace:** 刺激入力により、MCに痕跡 ($e_{slow}$) が残る。
2.  **Resonance:** 本能回路が反応し、Modulatorが発火 ($D(t)$ 上昇)。
3.  **Gating:** システムが共鳴状態に入り、痕跡を持つシナプスが強化される。