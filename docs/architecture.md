# Network Architecture & Topology

## 1. 領域マップ (ID Map)

ニューロンIDはフラットな配列上で以下のように割り当てられる。

| 領域 | ID範囲 | 役割 | 結合特性 |
| :--- | :--- | :--- | :--- |
| **Thinking (Input)** | `0` ~ `n_input - 1` | 感覚入力受容 | 出力: Fixed |
| **Thinking (Hidden)** | `n_input` ~ `n_input+n_hidden - 1` | 内部処理・特徴変換 | 入出力: Fixed |
| **Thinking (Motor)** | `...` ~ `n_think - 1` | 行動出力、学習ターゲット | 入力: Plastic (from Memory) |
| **Memory (Excitatory)** | `n_think` ~ `n_total - n_inh` | 文脈保持、リザーバ演算 | 出力: Plastic, 正の重み |
| **Memory (Inhibitory)** | `n_total - n_inh` ~ `n_total` | 発火抑制、安定化 | 出力: Plastic, **負の重み** |

※ `n_think` = `n_input` + `n_hidden` + `n_motor`
※ `n_inh` (抑制性ニューロン数) は 記憶野全体の 20% とする。

## 2. 結合トポロジー (Connectivity)

### A. Thinking Internal & Projection ($W_{fixed}$)
思考野内部および記憶野への情報の流れ（事前知識）。
* **Input $\to$ Hidden:** 入力特徴の変換（スパース結合）。
* **Hidden $\to$ Motor:** ベースラインとなる反射行動。
* **Thinking $\to$ Memory (Context Injection):** Input/Hidden からの情報を記憶野へブロードキャストする。

### B. Recurrent Reservoir ($W_{rec}$)
* **経路:** `Memory` $\leftrightarrow$ `Memory`
* **意味:** 入力情報を時間的に反響させ、短期記憶を形成する。
* **初期化:** スパース率 10%, 重みスケール調整済み。
* **制約:** **自己結合 ($i \to i$) は禁止（重み 0 固定）**。発火の暴走（Bursting）を防ぐため。
* **可塑性:** **あり** (SRGにより更新)

### C. Interface ($W_{inter}$)
* **経路:** `Memory` $\to$ `Hidden / Motor`
* **意味:** 過去の文脈に基づいて、思考野の処理（Hidden）や行動（Motor）にバイアスをかける。
* **初期化:** **0.0** (最初は接続なし)
* **可塑性:** **あり** (SRGにより学習)

## 3. 安定化機構 (Stabilization)
* **Dale's Law:** 抑制性ニューロンからの出力重みは常に負の値に固定される。
* **Global Weight Decay:** 学習中、可塑的結合（Plastic）は常にわずかに減衰し、使用されない記憶の忘却と重みの飽和を防ぐ。

## 4. 将来的な拡張・課題 (Future Work)
現在の実装（Phase 1）では見送られたが、スケーラビリティ確保のために将来的に検討すべき事項。

* **疎行列 (Sparse Matrix) の導入:** 現在の Dense 行列 ($O(N^2)$) はメモリ効率が悪いため、大規模化の際は `scipy.sparse` や隣接リスト方式へ移行する。
* **思考野への抑制性導入:** 現在、抑制性ニューロンは記憶野のみに存在するが、思考野内部の複雑な制御のために抑制性結合を導入する可能性がある。