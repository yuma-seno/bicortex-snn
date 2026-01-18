# Visual Encoding Specification

## 1. 視覚フロントエンド (CNN Backbone)
* **Model**: MobileNetV3-Small (Pre-trained on ImageNet)
* **Output Layer**: Final global average pooling layer (576-dim).
* **Role**: 生画像ピクセルから、高次な空間的・テクスチャ的特徴を抽出する「人工視神経」として機能。

## 2. SNNへの変換 (Current Mapping)
CNNの出力ベクトル $F \in \mathbb{R}^{576}$ を、SNNの入力電流 $I \in \mathbb{R}^{576}$ へ変換する。

1. **非負化 (ReLU)**: 生の出力には負の値が含まれるため、$\max(0, F)$ を適用する。
2. **エネルギー正規化**: 全入力ニューロンへの電流量の総和を $T_{sum} = 150.0$ に固定する。
   $$I_i = \frac{F_i}{\sum F} \cdot T_{sum}$$
3. **投影**: 各入力ニューロン $I_i$ は、記憶野 (Memory Cortex) の特定ニューロンへ疎結合 (5%) でランダムに接続される。

## 3. 既知の課題
* **解像度不足**: 記憶野のニューロン数が少ない場合、異なる視覚特徴（例：赤と青）の内部表現が重複し、誤った連想（汎化ミス）を引き起こす。解決策として、記憶野のスケールアップを適用する。