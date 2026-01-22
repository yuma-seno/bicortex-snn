# Visual Encoding Specification

## 1. 視覚フロントエンド (CNN Backbone)
* **Model**: MobileNetV3-Small (Pre-trained on ImageNet)
* **Output Layer**: Final global average pooling layer (576-dim).
* **Role**: 生画像ピクセルから、高次な空間的・テクスチャ的特徴を抽出する「人工視神経」として機能。

## 2. SNNへの変換 (Features to Currents)
CNNの出力ベクトル $F \in \mathbb{R}^{576}$ を、SNNの入力電流 $I \in \mathbb{R}^{576}$ へ変換するプロセス。

1. **特徴抽出**: 画像を入力し、576次元の特徴ベクトルを取得する。
2. **物理的分離 (Physical Separation)**:
    * 異なるクラス（例：赤と青）の特徴ベクトル間で、活動しているニューロンのインデックスが重複しないように処理する。
    * これにより、SNN内部での「混線（Cross-talk）」を物理的に防ぎ、学習の安定性を保証する。
    * *（注：将来的には、SNNの側抑制メカニズムによって自律的に分離されることが望ましいが、現段階では確実性を優先して前処理で行う）*
3. **正規化 (Normalization)**:
    * ベクトルのノルムを正規化し、入力強度のばらつきを抑える。
4. **ゲイン調整 (Gain Scaling)**:
    * SNNのニューロンを確実に発火させるのに十分な強度（例：Input Gain = 10.0 ~ 20.0）を掛ける。
