# Bi-Cortex SNN (BC-SNN)

**Bi-Cortex Spiking Neural Network: An Autonomous Real-Time Learning Architecture for Edge Devices**

![Status](https://img.shields.io/badge/Status-Phase%202%20(Visual)-orange)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 概要 (Overview)

**Bi-Cortex SNN (BC-SNN)** は、エッジデバイス上での自律的なリアルタイム適応を目指す、新しいアーキテクチャの Spiking Neural Network (SNN) 構想です。

従来の深層学習（誤差逆伝播法）に依存せず、**「意味的共鳴ゲーティング (Semantic Resonance Gating / SRG)」** と呼ばれる独自の局所則を用いることで、低演算コストでの文脈適応の実現を目指します。

### コア・コンセプト
本モデルは、単一のニューロン群を機能的に2つの領域（Cortex）に区分して動作します。

1.  **思考野 (Thinking Cortex / TC):**
    * **役割:** 高度な推論、世界認識、評価、反射。
    * **構造:** CNNやLLMなどの事前学習済みモデルをバックボーンとして利用します。
    * **特性:** パラメータは**固定 (Fixed)**。ここが「知能」の本体であり、運用中に再学習することはありません。
2.  **記憶野 (Memory Cortex / MC):**
    * **役割:** 短期記憶の保持、時間的文脈の統合。
    * **構造:** ランダムに結合されたリザーバ層。
    * **特性:** **内部結合が動的 (Plastic)** に変化する連想記憶バッファ。思考野の推論結果を一時的に保持・結合し、直近の状況に適応します。

**インターフェース (Interface):**
これら2つの領域を繋ぐ「シナプス結合」のことを指します。思考野の学習後、思考野の信号を記憶野へ適切にマッピングするために調整され、運用時は固定されます。

詳細な技術仕様については、[仕様書ドキュメント](docs/仕様書.md) を参照してください。

---

## 🚀 環境構築 (Getting Started)

本プロジェクトは **Dev Containers** (Docker) を用いた開発環境の構築を推奨しています。
お手持ちの GPU 環境（NVIDIA または AMD）に合わせて、適切なコンテナを選択して起動できます。

### 前提条件 (Prerequisites)
以下のツールがインストールされている必要があります。

1.  [Docker Desktop](https://www.docker.com/products/docker-desktop/) (または Docker Engine)
2.  [Visual Studio Code](https://code.visualstudio.com/)
3.  VS Code 拡張機能: [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

**※ AMD GPU (ROCm) 利用時の注意:**
Linux ホスト (Ubuntu 等) 上で、ホスト側に ROCm ドライバが正しくインストールされている必要があります。Windows (WSL2) での ROCm 利用は環境依存が激しいため、ネイティブ Linux 環境を推奨します。

### 起動手順 (Setup Steps)

1.  **リポジトリのクローン**
    ```bash
    git clone [https://github.com/your-username/bicortex-snn.git](https://github.com/your-username/bicortex-snn.git)
    cd bicortex-snn
    ```

2.  **VS Code で開く**
    ```bash
    code .
    ```

3.  **コンテナで再度開く**
    * VS Code 左下の緑色のアイコン（または `F1` キー）をクリックします。
    * コマンドパレットから **"Dev Containers: Reopen in Container"** を選択します。
    * **設定の選択:** 画面上部に構成の選択リストが表示されます。使用する GPU に合わせて選択してください。
        * `python-nvidia`: **NVIDIA GPU (CUDA)** を使用する場合
        * `python-amd`: **AMD GPU (ROCm)** を使用する場合

4.  **動作確認**
    コンテナが起動したら、ターミナルで以下のコマンドを実行し、GPU が認識されているか確認してください。
    ```bash
    python -c "import torch; print(f'Torch: {torch.__version__}, GPU: {torch.cuda.is_available()}')"
    ```

---

## 📂 ディレクトリ構成 (Structure)

```text
bicortex-snn/
├── .devcontainer/     # Dev Container 設定
│   ├── nvidia/        # CUDA環境用定義
│   ├── amd/           # ROCm環境用定義
│   └── entrypoint.sh  # 共通エントリーポイント
├── docs/              # 技術仕様書、数式定義
├── reports/           # 実験レポート、図表
├── experiments/       # 実験スクリプト
├── src/               # ソースコード
│   ├── core/          # ニューロンモデル、SNNエンジン (NumPy実装)
│   ├── envs/          # 実験環境 (GridWorld, Games)
│   └── utils/         # 可視化ツール、ロガー
├── tests/             # 単体テスト
├── requirements.txt   # Python依存ライブラリ
└── README.md          # 本ファイル

```

---

## 🛠️ 現在の状況について

ロードマップ及び現在の状況は [作業一覧_チェックリスト.md](docs/作業一覧_チェックリスト.md) を参照してください。

---