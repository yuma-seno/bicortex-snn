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
これら2つの領域を繋ぐ「シナプス結合」のことを指します。思考野の信号を記憶野へ適切にマッピングするために事前に調整され、運用時は固定されます。

### 動作原理
BC-SNNにおける「学習」とは、長期的なスキルの獲得ではなく、**「短期的な文脈の形成 (Context Formation)」** を指します。

* **Innate Intelligence (先天的な知能):** 思考野は最初から世界を理解しています（例：画像認識、言語理解）。
* **Trace & Link (痕跡と結合):** 記憶野は、思考野から送られてきた情報を一時的に保持（Trace）し、報酬信号（ドーパミン）に基づいてそれらを結びつけます（Link）。
* **Transient Adaptation (一時的な適応):** 形成された結合は時間とともに減衰し、常に最新の状況に適応し続けます。

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
├── docs/              # 技術仕様書、数式定義、実験レポート
├── src/               # ソースコード
│   ├── core/          # ニューロンモデル、SNNエンジン (NumPy実装)
│   ├── envs/          # 実験環境 (GridWorld, Games)
│   └── utils/         # 可視化ツール、ロガー
├── tests/             # 単体テスト
├── notebooks/         # 実験用 Jupyter Notebooks
├── requirements.txt   # Python依存ライブラリ
└── README.md          # 本ファイル

```

---

## 🗺️ ロードマップ (Roadmap)

現在の開発フェーズは **Phase 1** です。

* [x] **Phase 0: Project Setup & Preparation**
* リポジトリ管理、Dev Containers 環境構築、基礎ドキュメント整備。

* [x] **Phase 1: Proof of Concept** (Core Logic Verification)
* 基本的なLIFモデルと共鳴学習則(SRG)の実装・検証。


* [ ] **Phase 2: Visual Integration**
* CNNバックボーン(MobileNetV3)の統合と空間ナビゲーション。


* [ ] **Phase 3: Real-time Adaptation**
* ゲーム環境での動的なルール変化への適応テスト。


* [ ] **Phase 4: Spiking LLM Experiment**
* 思考野への Spiking LLM (RWKV等) 統合実験。



詳細な進捗は [作業一覧_チェックリスト.md](docs/作業一覧_チェックリスト.md) (またはプロジェクト管理ツール) を参照してください。

---