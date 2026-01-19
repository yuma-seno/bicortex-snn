# Network Architecture & Topology

## 1. 領域マップ (ID Map)
* **思考野 (Thinking Cortex / TC):** $0 \dots N_{th}-1$
* **記憶野 (Memory Cortex / MC):** $N_{th} \dots N_{total}-1$

## 2. 結合トポロジー (Connectivity)

### A. Thinking Internal & Projection ($W_{fixed}$)
* **経路:** `Input` $\to$ `Hidden` $\to$ `Motor` および `Thinking` $\to$ `Memory`
* **可塑性:** **なし (Fixed)**。
* **役割:** 事前学習済みモデル（CNN等）の特徴抽出能力と、文脈注入（Context Injection）を担う。

### B. Recurrent Reservoir ($W_{rec}$)
* **経路:** `Memory` $\leftrightarrow$ `Memory`
* **意味:** 記憶野内部での情報の結びつき（連想）や、時間的な反響を形成する。
* **初期化:** 全結合（ただし初期重みは 0.0 または微弱）。
* **可塑性:** **あり (Plastic)**。
    * **本アーキテクチャの核となる学習領域。**
    * **3要素学習則 (3-Factor Learning):**
        1.  **Presynaptic Activity:** 入力側の発火（またはトレース）。
        2.  **Postsynaptic Activity:** 出力側の発火。
        3.  **Neuromodulator (Reward):** ドーパミン等の報酬信号による強化・抑制。

### C. Interface ($W_{inter}$)
* **経路:** `Memory` $\to$ `Hidden / Motor` (Readout)
* **意味:** 記憶野の特定のパターンが、どのような思考・行動を誘発するかを定義する。
* **可塑性:** **条件付き可塑 (Conditionally Plastic)**。
    * 基本設計としては固定だが、強化学習（RL）タスクにおいては、誤った行動を抑制するために可塑性を持たせることがある。

## 3. 安定化機構 (Stabilization)
* **Neural Adaptation (順応):** ニューロンの発火ごとの閾値上昇による恒常性維持。内部結合が強化されても、無限の発火暴走（Bursting）を防ぐ安全装置として機能する。
* **Surgical Wiring (初期化時):** 不要なショート（短絡）を防ぐため、学習対象以外のランダムな背景結合は初期状態では切断しておく。

## 4. 将来的な拡張・課題 (Future Work)
（変更なし）