# bdi_agent_for_aiwolf

人狼知能コンテスト（自然言語部門） のLLMを用いたサンプルエージェントです。

## 環境構築

> [!IMPORTANT]
> Python 3.11以上が必要です。

```bash
git clone https://github.com/iggy157/bdi_agent_for_aiwolf.git
cd bdi_agent_for_aiwolf
cp config/config.yml.example config/config.yml
cp config/.env.example config/.env
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 実行方法・その他

/config/.envにgoogleかopenaiのapiキーを設定します。
/config/config.yml/llmにどちらのapiを使うのか、sleep timeはどれくらいに設定するのかを記述します。(推奨sleep time:googleのとき3, openaiのとき0)
事前に[サーバー](https://github.com/aiwolfdial/aiwolf-nlp-server)で5人または13人用のサーバーを立ち上げます。
サーバーを立ち上げたのち、python src/main.pyによって人狼ゲームの自己対戦を実行できます。

実行方法の詳細やその他の情報については[aiwolf-nlp-agent](https://github.com/aiwolfdial/aiwolf-nlp-agent)をご確認ください。

# bdi_agent_for_aiwolf/judgement_werewolf/libsvm

[詳細はこちら](/judgement_werewolf/libsvm/README.md)

```bash
# 役職情報,発話内容の形式にログを整形
python judgement_werewolf/libsvm/log_formatter_player_split.py
# numpyインストール
pip install numpy
# scikit-learnインストール
pip install scikit-learn
# libsvm形式に変換
python judgement_werewolf/libsvm/run_all_embeddings.py
# pandasインストール
pip install pandas
# matplotlib
pip install matplotlib
# seabornインストール
pip install seaborn
# 訓練
python judgement_werewolf/libsvm/train_werewolf_models.py
```
/judgement_werewolf/libsvm/models/にモデルや結果が保存されます。
ベストモデル(/judgement_werewolf/libsvm/models/Word2Vec/best_model_word2vec.joblib)を呼び出して使用してください。


# aiwolf-nlp-server

人狼知能コンテスト（自然言語部門） のゲームサーバです。
(https://github.com/aiwolfdial/aiwolf-nlp-server)

## 実行方法

デフォルトのサーバアドレスは `ws://127.0.0.1:8080/ws` です。エージェントプログラムの接続先には、このアドレスを指定してください。\
同じチーム名のエージェント同士のみをマッチングさせる自己対戦モードは、デフォルトで有効になっています。そのため、異なるチーム名のエージェント同士をマッチングさせる場合は、設定ファイルを変更してください。\
設定ファイルの変更方法については、[設定ファイルについて](./doc/config.md)を参照してください。

### Linux

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-linux-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-linux-amd64
./aiwolf-nlp-server-linux-amd64 -c ./default_5.yml # 5人ゲームの場合
# ./aiwolf-nlp-server-linux-amd64 -c ./default_13.yml # 13人ゲームの場合
```

### Windows

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-windows-amd64.exe
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
.\aiwolf-nlp-server-windows-amd64.exe -c .\default_5.yml # 5人ゲームの場合
# .\aiwolf-nlp-server-windows-amd64.exe -c .\default_13.yml # 13人ゲームの場合
```

### macOS (Intel)

> [!NOTE]
> 開発元が不明なアプリケーションとしてブロックされる場合があります。\
> 下記サイトを参考に、実行許可を与えてください。  
> <https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac>

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-darwin-amd64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-darwin-amd64
./aiwolf-nlp-server-darwin-amd64 -c ./default_5.yml # 5人ゲームの場合
# ./aiwolf-nlp-server-darwin-amd64 -c ./default_13.yml # 13人ゲームの場合
```

### macOS (Apple Silicon)

> [!NOTE]
> 開発元が不明なアプリケーションとしてブロックされる場合があります。\
> 下記サイトを参考に、実行許可を与えてください。  
> <https://support.apple.com/ja-jp/guide/mac-help/mh40616/mac>

```bash
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/aiwolf-nlp-server-darwin-arm64
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_5.yml
curl -LO https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/default_13.yml
curl -Lo .env https://github.com/aiwolfdial/aiwolf-nlp-server/releases/latest/download/example.env
chmod u+x ./aiwolf-nlp-server-darwin-arm64
./aiwolf-nlp-server-darwin-arm64 -c ./default_5.yml # 5人ゲームの場合
# ./aiwolf-nlp-server-darwin-arm64 -c ./default_13.yml # 13人ゲームの場合
```
