# Bi-Cortex SNN (BC-SNN)

**Bi-Cortex Spiking Neural Network: An Autonomous Real-Time Learning Architecture for Edge Devices**

![Status](https://img.shields.io/badge/Status-Phase%201%20(PoC)-blue)
![Python](https://img.shields.io/badge/Python-3.10%2B-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 📖 概要 (Overview)

**Bi-Cortex SNN (BC-SNN)** は、エッジデバイス上での自律的なリアルタイム学習と適応を目指す、新しいアーキテクチャの Spiking Neural Network (SNN) です。

従来の深層学習（誤差逆伝播法）に依存せず、**「意味的共鳴ゲーティング (Semantic Resonance Gating)」** と呼ばれる独自の局所学習則を用いることで、低演算コストでの文脈学習を実現します。

### コア・コンセプト
本モデルは、単一のニューロン群を機能的に2つの領域（Cortex）に区分して動作します。

1.  **思考野 (Thinking Cortex / TC):**
    * **役割:** 特徴抽出、概念認識、即時的な反射行動。
    * **特性:** パラメータは**固定 (Fixed)**。CNNやLLMなどの事前学習済みモデルの知識を流用します。
2.  **記憶野 (Memory Cortex / MC):**
    * **役割:** 短期記憶の保持、時間的文脈の統合。
    * **特性:** パラメータは**動的 (Plastic)**。思考野が「意味」を検知した瞬間のみ可塑性が活性化し、因果関係を学習します。

### 主な特徴
* **Layer-less Architecture:** 物理的な層構造を持たず、フラットなニューロン配列とIDマップによって構造を定義。
* **Dual Traces:** 「即時トレース」と「適格性トレース(Eligibility Trace)」を併用し、数秒〜数十秒のラグがある事象間の因果関係を学習可能。
* **Edge Native:** バックプロパゲーション不要のため、マイコンやFPGAなどのリソース制約環境でも動作可能（予定）。

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