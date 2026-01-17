# Mathematical Model & Implementation Details

## 1. Neuron Model (LIF: Leaky Integrate-and-Fire)

計算コストを最小化するため、離散時間における積分発火モデルを採用しています。

### 数式定義
$$v_i(t) = v_i(t-1) \cdot \alpha + I_{ext}(t) + \sum_{j} w_{ij} x_{fast, j}(t-1)$$

### コード上の変数対応 (`src/core/engine.py`)

| 数式記号 | 変数名 | 説明 | 設定値(例) |
| :--- | :--- | :--- | :--- |
| $v_i(t)$ | `self.v` | 膜電位 (Membrane Potential) | 初期値 0.0 |
| $\alpha$ | `self.alpha` | 減衰係数 ($\exp(-dt/\tau_m)$) | $\tau_m=20ms \to \approx 0.95$ |
| $I_{ext}(t)$ | `input_current` | 外部入力電流 | - |
| $w_{ij}$ | `self.W` | 結合荷重行列 | - |
| $v_{th}$ | `self.v_th` | 発火閾値 | 1.0 |

---

## 2. Dual Traces (2つのトレース変数)

時間的信用割当問題（Temporal Credit Assignment）を解決するため、時定数の異なる2つのトレース変数を保持します。

### A. Immediate Trace (即時トレース)
シナプス後電流 (PSC) の近似として使用されます。

* **数式:** $x_{fast}(t) = x_{fast}(t-1) \cdot \alpha_{fast} + S(t)$
* **変数名:** `self.x_fast`
* **時定数:** `self.tau_fast` = 20.0 ms
* **用途:** 通常の信号伝達 ($W \cdot x_{fast}$)

### B. Eligibility Trace (適格性トレース)
因果関係の学習（SRG）に使用される「発火の残り香」です。

* **数式:** $e_{slow}(t) = e_{slow}(t-1) \cdot \alpha_{slow} + S(t)$
* **変数名:** `self.e_slow`
* **時定数:** `self.tau_slow` = 2000.0 ms (2秒)
* **用途:** 学習時のプレニューロン項 ($\Delta W \propto Post \cdot Pre_{slow}$)

---

## 3. Refractory Period (不応期)

発火直後のニューロンは一定時間再発火できません。

* **変数名:** `self.refractory_count`
* **期間:** 2.0 ms
* **動作:** カウントが 0 になるまで $v(t)$ を強制的に 0 にリセットし続ける。