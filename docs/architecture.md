# Network Architecture & Topology

## 1. 領域マップ (ID Map)

ニューロンIDはフラットな配列上で以下のように割り当てられる。

| 領域 | ID範囲 | 役割 | 結合特性 |
| :--- | :--- | :--- | :--- |
| **Thinking (Concept)** | `0` ~ `n_input - 1` | 入力受容、特徴抽出 | 出力: Fixed |
| **Thinking (Motor)** | `n_input` ~ `n_think - 1` | 行動出力、学習ターゲット | 入力: Plastic (from Memory) |
| **Memory (Excitatory)** | `n_think` ~ `n_total - n_inh` | 文脈保持、リザーバ演算 | 出力: Plastic, 正の重み |
| **Memory (Inhibitory)** | `n_total - n_inh` ~ `n_total` | 発火抑制、安定化 | 出力: Plastic, **負の重み** |

※ `n_think` = `n_input` + `n_motor`
※ `n_inh` (抑制性ニューロン数) は 記憶野全体の 20% とする。

## 2. 結合トポロジー (Connectivity)

### A. Context Injection ($W_{fixed}$)
* **経路:** `Thinking(Concept)` $\to$ `Memory`
* **意味:** 「今、何が見えているか」を記憶野に注入する。
* **初期化:** スパース率 20%, 重み $U(0, 0.5)$

### B. Recurrent Reservoir ($W_{rec}$)
* **経路:** `Memory` $\leftrightarrow$ `Memory`
* **意味:** 入力情報を時間的に反響させ、短期記憶を形成する。
* **初期化:** スパース率 10%, 重み $U(0, 0.5)$ (抑制性は $\times -2.0$)
* **制約:** **自己結合 ($i \to i$) は禁止（重み 0 固定）**。発火の暴走（Bursting）を防ぐため。
* **可塑性:** **あり** (SRGにより更新)

### C. Interface ($W_{inter}$)
* **経路:** `Memory` $\to$ `Thinking(Motor)`
* **意味:** 過去の文脈に基づいて行動を誘発する。
* **初期化:** **0.0** (最初は接続なし)
* **可塑性:** **あり** (SRGにより学習)

## 3. 安定化機構 (Stabilization)
* **Dale's Law:** 抑制性ニューロンからの出力重みは常に負の値に固定される。
* **Masking:** 学習時、`mask_plastic` が `True` の箇所のみ重み更新を適用する。