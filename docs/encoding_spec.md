# Visual Encoding Specification

## 1. 視覚フロントエンド (CNN Backbone)
* **Model**: MobileNetV3-Small (Pre-trained on ImageNet)
* **Output Layer**: Final global average pooling layer (576-dim).
* **Role**: 生画像ピクセルから、高次な空間的・テクスチャ的特徴を抽出する「人工視神経」として機能。

## 2. SNNへの変換 (Current Mapping)
CNNの出力ベクトル $F \in \mathbb{R}^{576}$ を、SNNの入力電流 $I \in \mathbb{R}^{576}$ へ変換する。

1. **非負化 (ReLU)**: 生の出力には負の値が含まれるため、$\max(0, F)$ を適用する。
2. **差分強調 (Pattern Separation)**: 赤/青など対照刺激がある場合は共通成分を除去し、$F' = \max(0, F - \bar{F})$ を用いる。
3. **競合抑制 (Contrast Enhancement)**: 特徴間の二乗差分を取り、さらに `k-WTA` (k-Winner-Take-All) ライクなスパース化処理を適用して、上位10%の強い特徴のみを残す。
4. **エネルギー正規化**: 全入力ニューロンへの電流量の総和を $T_{sum} = 150.0$ に固定する。
   $$I_i = \frac{F''_i}{\sum F''} \cdot T_{sum}$$
5. **投影**: 各入力ニューロン $I_i$ は、記憶野 (Memory Cortex) の特定ニューロンへ**極めて疎な結合 (0.2%)** でランダムに接続される。
   * **結合重み:** $w_{vis} = 6.0$ (高いゲインで少数のニューロンを強く駆動する)

## 3. 既知の課題と解決策
* **パターンの重複**: 単純なランダム投影では類似した画像特徴が混ざり合う。
* **解決策**: 入力段階での強力なスパース化（Sparsification）と、強化学習（誤発火への罰則）を組み合わせることで、記憶野内での表現分離を達成した。