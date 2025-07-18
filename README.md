# 🧠 bdi\_agent\_for\_aiwolf

人狼知能コンテスト（自然言語部門）向けの、LLMを用いたサンプルエージェントです。

---

## ⚙️ 環境構築

> **ℹ️ Python 3.11以上が必要です**

```bash
git clone https://github.com/iggy157/bdi_agent_for_aiwolf.git
cd bdi_agent_for_aiwolf
cp config/config.yml.example config/config.yml
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## 🧪 機械学習の実行

以下の準備が完了していることを前提とします：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

準備ができたら、以下のステップでモデルを訓練してください：

```bash
# ログ整形（役職・発話内容の形式化）
python judgement_werewolf/libsvm/log_formatter_player_split.py

# 必要ライブラリのインストール
pip install numpy
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install seaborn

# libsvm形式に変換
python judgement_werewolf/libsvm/run_all_embeddings.py

# モデル訓練
python judgement_werewolf/libsvm/train_werewolf_models.py
```

---

## 📁 モデルファイル（`.joblib`）

| ファイル名                                  | 内容                                                |
| -------------------------------------- | ------------------------------------------------- |
| `best_model_{embedding_type}.joblib`   | 最も性能の良かったモデル（パイプラインごと）                            |
| `{model_name}_{embedding_type}.joblib` | 各モデルごとの訓練済みモデル（例: `randomforest_word2vec.joblib`） |

---

## 📄 評価レポート（`.txt`）

| ファイル名                                   | 内容                            |
| --------------------------------------- | ----------------------------- |
| `training_results_{embedding_type}.txt` | 各モデルの訓練結果、パラメータ、評価スコア、分類レポート等 |

---

## 📊 可視化プロット（`.png`）

| ファイル名                                   | 内容                          |
| --------------------------------------- | --------------------------- |
| `model_comparison_{embedding_type}.png` | モデルごとのF1スコア・Accuracyの比較棒グラフ |
| `confusion_matrix_{embedding_type}.png` | 最良モデルにおける混同行列ヒートマップ         |

---

## 💡 埋め込み手法ごとの違い

以下の3種類の埋め込み手法に対応し、それぞれに対して上記のファイルが生成されます：

* Word2Vec
* FastText
* BERT

---

## ✅ モデルの使用

以下のように、ベストモデルを呼び出して使用できます：

```
judgement_werewolf/libsvm/models/Word2Vec/best_model_word2vec.joblib
```

---

## 🚀 実行方法・その他

1. `/config/.env` に Google または OpenAI の APIキーを設定します。

2. `/config/config.yml` の `llm` セクションに、使用するAPI種別（`google`または`openai`）と `sleep_time` を指定します。

   * 推奨値: `google` の場合は 3、`openai` の場合は 0

3. [サーバー](https://github.com/aiwolfdial/aiwolf-nlp-server) を起動して、5人または13人用のゲームに対応させます。

4. サーバー起動後、以下のコマンドで自己対戦が可能です：

```bash
python src/main.py
```

> 詳細は [aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent) を参照してください。

---

## 📂 judgement\_werewolf/libsvm の詳細

プログラムの動作や説明について詳しくは以下のREADMEをご確認ください：

👉 [詳細はこちら](/judgement_werewolf/libsvm/README.md)

---

# 🎮 aiwolf-nlp-server

人狼知能コンテスト（自然言語部門）向けのゲームサーバです。
リポジトリ：[https://github.com/aiwolfdial/aiwolf-nlp-server](https://github.com/aiwolfdial/aiwolf-nlp-server)

---

## 🏁 実行方法

* デフォルトのサーバアドレス: `ws://127.0.0.1:8080/ws`
* 自己対戦モードはデフォルトで有効（同一チーム名のエージェントのみ対戦）

> ⚙️ 異なるチーム名で対戦したい場合は、設定ファイルを編集してください → [設定ファイルについて](./doc/config.md)

---

### 🐧 Linux

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-linux-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-linux-amd64
./aiwolf-nlp-server-linux-amd64 -c ./default_5.yml # 5人ゲーム
# ./aiwolf-nlp-server-linux-amd64 -c ./default_13.yml # 13人ゲーム
```

---

### 🪟 Windows

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-windows-amd64.exe
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
.\aiwolf-nlp-server-windows-amd64.exe -c .\default_5.yml # 5人ゲーム
# .\aiwolf-nlp-server-windows-amd64.exe -c .\default_13.yml # 13人ゲーム
```

---

### 🍎 macOS (Intel)

> **※注意：** 実行時に「開発元不明」の警告が出る場合は以下を参考に許可してください。
> [https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac](https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac)

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-darwin-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-darwin-amd64
./aiwolf-nlp-server-darwin-amd64 -c ./default_5.yml # 5人ゲーム
# ./aiwolf-nlp-server-darwin-amd64 -c ./default_13.yml # 13人ゲーム
```

---

### 🍏 macOS (Apple Silicon)

> **※注意：** 実行時に「開発元不明」の警告が出る場合は以下を参考に許可してください。
> [https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac](https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac)

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-darwin-arm64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-darwin-arm64
./aiwolf-nlp-server-darwin-arm64 -c ./default_5.yml # 5人ゲーム
# ./aiwolf-nlp-server-darwin-arm64 -c ./default_13.yml # 13人ゲーム
```

---
