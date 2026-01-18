# Network Architecture & Topology

## 1. 領域マップ (ID Map)
（変更なし）

## 2. 結合トポロジー (Connectivity)

### A. Thinking Internal & Projection ($W_{fixed}$)
（変更なし）

### B. Recurrent Reservoir ($W_{rec}$)
* **経路:** `Memory` $\leftrightarrow$ `Memory`
* **意味:** 記憶野内部での情報の結びつき（連想）や、時間的な反響を形成する。
* **初期化:** 全結合（ただし初期重みは 0.0 または微弱）。
* **可塑性:** **あり (Plastic)**。
    * **本アーキテクチャの核となる学習領域。**
    * 異なるイベント（例：ベルとエサ）に対応するニューロングループ間の結合を、SRG学習則によって強化することで「連想記憶」を形成する。

### C. Interface ($W_{inter}$)
* **経路:** `Memory` $\to$ `Hidden / Motor` (Readout)
* **意味:** 記憶野の特定のパターンが、どのような思考・行動を誘発するかを定義する。
* **初期化:** 事前定義された固定パターン（例：エサの記憶→ヨダレ）。
* **可塑性:** **なし (Fixed)**。
    * 出力の解釈（Readout）は固定し、内部のダイナミクス変化によって出力を制御する。

## 3. 安定化機構 (Stabilization)
* **Neural Adaptation (順応):** ニューロンの発火ごとの閾値上昇による恒常性維持。内部結合が強化されても、無限の発火暴走（Bursting）を防ぐ安全装置として機能する。
* **Surgical Wiring (初期化時):** 不要なショート（短絡）を防ぐため、学習対象以外のランダムな背景結合は初期状態では切断しておく。

## 4. 将来的な拡張・課題 (Future Work)
（変更なし）