# Visual Encoding Specification

## 1. 視覚フロントエンド (CNN Backbone)
* **Model**: MobileNetV3-Small (Pre-trained on ImageNet)
* **Output Layer**: Final global average pooling layer (576-dim).
* **Role**: 生画像ピクセルから、高次な空間的・テクスチャ的特徴を抽出する「人工視神経」として機能。このモジュールは **Thinking Cortex** に属し、パラメータは固定される。

## 2. Interfaceによる変換 (Signal Adaptation)
Thinking Cortexの出力 $F \in \mathbb{R}^{576}$ は、**Interface結合** を経由して Memory Cortex への入力電流 $I$ に変換される。

この変換特性は、システムのセットアップ段階（Phase 2: Interface Calibration）で決定され、運用時は固定される。

### 変換プロセス
1.  **特徴抽出**: 画像を入力し、576次元の特徴ベクトルを取得する。
2.  **物理的分離 (Physical Separation)**:
    * **Contrastive Subtraction**: キャリブレーション時に、対立する概念（例：赤 vs 青）の特徴ベクトル同士を引き算する。
    * **直交化**: 共通して反応してしまう成分（ノイズ）を物理的に除去し、ベクトルを直交化させる。これより、SNN内部での「混線（Cross-talk）」を完全に防ぐ。
3.  **正規化 (Normalization)**:
    * ベクトルのノルムを正規化し、入力強度のばらつきを抑える。
4.  **ゲイン調整 (Gain Scaling)**:
    * 高い閾値を持つSNNニューロンを確実に発火させるため、強力なゲイン（例：Input Gain = 50.0）を適用する。

## 3. 記憶野への投影 (Projection to Memory)
* **構造化投影**:
    * Interface結合は、上記で分離・強調された特徴ベクトルを、Memory Cortex の対応するニューロン群へ投影する。
    * これにより、入力パターンのわずかな差異が、記憶野上では明確に異なる発火パターンとして表現される。