# Network Architecture & Topology

## 1. 領域マップ (ID Map)
物理的なニューロンは以下の2領域のみで構成される。

* **思考野 (Thinking Cortex / TC):** $0 \dots N_{th}-1$
    * **構成要素:** Sensory (感覚), Hidden (中間・調節), Motor (運動)。
    * **役割:** **推論、世界モデルの保持、評価（Modulatorによるドーパミン生成）。**
    * **特性:** 固定 (Fixed)。事前トレーニングによって構築された「不変の知能」。
* **記憶野 (Memory Cortex / MC):** $N_{th} \dots N_{total}-1$
    * **役割:** **短期記憶の保持、一時的な文脈の結合 (Temporary Association)。**
    * **特性:** 可塑 (Plastic)。直近の出来事の因果関係を一時的に記録するために変化する。

## 2. 結合トポロジー (Connectivity)

### A. Thinking Internal ($W_{fixed}$)
* **経路:** `Input` $\to$ `Hidden` $\to$ `Motor` (思考野内部)
* **実体:** 事前学習済みの重み行列。
* **役割:** 入力に対する解釈や推論、および**本能的な報酬評価（Modulatorの発火）**はここで行われる。

### B. The Interface ($W_{inter}$)
* **経路:** `Thinking` $\leftrightarrow$ `Memory` (領域間の相互結合)
* **実体:** 思考野ニューロンと記憶野ニューロンを直接結ぶシナプス結合。
    * **Projection ($TC \to MC$):** 思考野の信号を記憶野へ書き込む。**固定 (Fixed)**。
    * **Readout ($MC \to TC$):** 記憶野の痕跡を行動へ変換する。**可塑 (Plastic)**。
* **役割:** 思考野で処理された「意味」を記憶野へ保存し、記憶野の「文脈」を行動へ反映させるパス。

### C. Recurrent Reservoir ($W_{rec}$)
* **経路:** `Memory` $\leftrightarrow$ `Memory` (記憶野内部)
* **実体:** 記憶野ニューロン同士のリカレント結合。
* **可塑性:** **あり (Plastic / Real-time)**。
    * **役割:** **「痕跡の保持と結合 (Trace Binding)」**。
    * 過去の事象（Trace）と現在の事象（Activity/Reward）を、報酬信号（Modulator活動）をトリガーとして一時的に結びつける。

## 3. 安定化機構 (Stabilization)
* **Synaptic Decay (忘却):** 記憶野関連の結合（$W_{rec}, W_{readout}$）は、強化され続けない限り自然に減衰する。
* **State Reset (状態リセット):** タスクの区切りでトレースをリセットする。

## 4. メカニズムの要点
* **推論 (Reasoning):** 思考野が担当。入力から最適な解を導き出す。
* **文脈 (Context):** 記憶野とInterface(Readout)が担当。思考野が出した評価（報酬）に基づき、直前の知覚情報を行動と結びつける。