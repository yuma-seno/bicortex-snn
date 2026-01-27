# Network Architecture & Topology

## 1. 領域マップ (ID Map)
物理的なニューロンは単一の配列上に配置され、ID範囲によって役割が区分される。

| 領域名 | サブセット | 役割 | 特性 |
| :--- | :--- | :--- | :--- |
| **Thinking Cortex (TC)** | **Sensory** | 外部センサー入力 | 固定 (Fixed) |
| | **Concept** | 特徴抽出・意味認識 (**SRG監視対象**) | 固定 (Fixed) |
| | **Motor** | 行動生成・出力 | 固定 (Fixed) |
| **Memory Cortex (MC)** | - | 文脈保持・リザーバ演算 | **可塑 (Plastic)** |

## 2. 記憶野 (Memory Cortex) の構成
* **Reservoir Computing:** ランダムかつスパースに結合されたリカレント回路。
* **E/I Balance (Dale's Law):**
    * **Exc (興奮性):** 80% (出力重みが正)
    * **Inh (抑制性):** 20% (出力重みが負)
    * これにより記憶活動の爆発を防ぎ、安定した「残響」を生成する。

## 3. 結合トポロジーと情報の流れ

### 3.1 固定結合 (Fixed / Pre-calibrated)
これらは初期化時またはキャリブレーション時に設定され、SRGによる更新を受けない。

* **Thinking Cortex:**
    * $TC_{sensory} \to TC_{concept}$
    * $TC_{concept} \to TC_{motor}$
* **Interface:**
    * **$TC_{concept} \to MC$ (Injection):** 概念を記憶野へ投影する。
    * **$MC \to TC_{concept}$ (Recall):** 記憶活動を概念へフィードバックする。

### 3.2 可塑的結合 (Plastic / SRG Target)
SRG（意味的共鳴ゲーティング）によってリアルタイムに重みが更新される結合。

* **$MC \to MC$ (Recurrent):**
    * 記憶野内部の結合。
    * 過去の記憶状態（Pre）と現在の記憶状態（Post）の因果関係を学習する。
    * これにより、「ベルの記憶状態」が自律的に「エサの記憶状態」へ遷移するようになる。