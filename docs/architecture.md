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

### 3.1 固定結合 (Fixed / Pre-trained)
これらは初期化時または学習済みモデルのロード時に設定され、SRGによる更新を受けない。

* **$TC_{sensory} \to TC_{concept}$:** 入力を意味的特徴へ変換する（CNNバックボーン等）。
* **$TC_{concept} \to TC_{motor}$:** 基本的な反射行動や推論結果を出力する。
* **$TC_{concept} \to MC$ (Injection):** 現在認識している「概念」を記憶野へブロードキャストする。

### 3.2 可塑的結合 (Plastic / SRG Target)
SRG（意味的共鳴ゲーティング）によってリアルタイムに重みが更新される結合。

1.  **$MC \to TC_{motor}$ (Interface):**
    * **最重要結合。**
    * 「過去の文脈($MC$)」を「現在の行動($Motor$)」に直結させる。
    * これにより、思考野が反射的に行った行動が、その直前の文脈と紐づけられる（条件付け）。
2.  **$MC \to MC$ (Recurrent):**
    * 短期記憶の保持時間を調整するために微調整される。