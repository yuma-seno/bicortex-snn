# Visual Encoding Specification

## 1. 視覚フロントエンド (CNN Backbone)
* **Model**: MobileNetV3-Small (Pre-trained on ImageNet)
* **Output Layer**: Final global average pooling layer (576-dim).
* **Role**: 生画像ピクセルから、高次な空間的・テクスチャ的特徴を抽出する「人工視神経」として機能。このモジュールは **Thinking Cortex** に属し、パラメータは固定される。

## 2. 変換プロセス (Encoding Process)
`src/core/visual.py` の実装に基づき、入力画像は以下の標準的なパイプラインで処理され、SNNへの入力電流 $I$ に変換される。

### 手順
1.  **前処理 (Preprocessing)**:
    * ImageNet学習済みモデルの標準仕様に基づく正規化（Normalize）、リサイズ、クロップを行う。
2.  **特徴抽出 (Feature Extraction)**:
    * CNNバックボーン（`features` および `avgpool`）を通過させ、最終的な特徴マップを取得する。
3.  **平坦化 (Flatten)**:
    * テンソルを平坦化し、**576次元**の特徴ベクトルを出力する。
    * 値の範囲はReLU活性化関数通過後の非負の値（概ね 0.0 ～ 6.0 程度）となり、これをそのままニューロンへの入力電流として利用する。

## 3. 記憶野への投影 (Projection to Memory)
* **構造化投影**:
    * 抽出された576次元の特徴ベクトルは、**Interface結合** を経由して Memory Cortex へ投影される。
    * ※ Phase 2の実装段階において、この高次元ベクトルを疎行列 (Sparse Matrix) 等を用いて効率的に記憶野へ接続する手法が適用される。