# 🧠 Werewolf LLM Judgment Pipeline - README

本プロジェクトは、自然言語形式の発話ログからプレイヤーの「人狼/村人」判定を行う機械学習モデルを構築する一連の処理スクリプト群です。各ステップはログ整形、特徴抽出、埋め込み生成、そしてモデル訓練の4段階に分かれています。

---

## 📂 構成概要

### 1. `log_formatter.py` - プレイヤー別ログ整形

* 元の人狼ログ（`.log`）を加工し、\*\*プレイヤー単位のテキストファイル（`.txt`）\*\*へ分割保存します。
* 発話行から、話者の役職（人狼 or 村人）を基にラベル（1:人狼, 0:村人）を付与。
* プレイヤーごとにラベル付き発話行を出力。
* 出力形式：

  ```
  <ラベル>,<発話内容>
  ```

---

### 2. `{bert, fasttext, word2vec}_processor.py` - 埋め込み生成スクリプト

#### 共通点

* `datasets/data/` 配下の各 `.txt` ファイルを読み込み、文単位で特徴ベクトルを生成。
* 各発話文を `libsvm` 形式に変換し保存。
* ラベルは `+1`（人狼） または `-1`（村人）として出力。
* 出力形式例（libsvm）:

  ```
  1 1:0.234 2:0.134 ... 768:0.556
  ```

#### 各埋め込みの特徴

| ファイル名                   | 手法       | 特徴                                              |
| ----------------------- | -------- | ----------------------------------------------- |
| `bert_processor.py`     | BERT     | 事前学習済みの日本語BERTを利用して `[CLS]` ベクトルを抽出（代替TF-IDFあり） |
| `fasttext_processor.py` | FastText | FastTextで訓練した単語ベクトルの平均（代替文字N-gram + TF-IDFあり）   |
| `word2vec_processor.py` | Word2Vec | Word2Vecで訓練した単語ベクトルの平均（代替TF-IDFあり）              |

---

### 3. `train_werewolf_models.py` - 高精度モデル訓練（+ 可視化）

* 各プレイヤー単位のlibsvmファイルを集約し、発話ベクトルをプレイヤー単位で統計的に集約（平均、標準偏差など）。
* ラベルは**多数決によりプレイヤー単位で1 or 0に決定**。
* `RandomForest`, `GradientBoosting`, `SVM`, `LogisticRegression` を訓練。
* 5分割交差検証 + グリッドサーチにより最良モデルを選出。
* 結果ファイルの保存：

  * `.joblib`: 訓練済みモデル（全モデル + ベストモデル）
  * `.txt`: 評価結果（各モデルのスコア・分類レポート）
  * `.png`: モデル比較プロット（F1スコア・Accuracy）、混同行列

---

### 4. `train_remaining_models.py` - 軽量モデル訓練（FastText / BERT向け）

* 精度よりも**訓練速度重視**で、簡易パラメータ設定（GridSearchなし）。
* 上記スクリプトと同様にlibsvm形式からプレイヤー特徴量を集約し、モデルを訓練・保存します。

---

## 🗂️ 出力ファイル構成例

```
judgement_werewolf/
├── libsvm/
│   ├── datasets/
│   │   ├── data/                         # log_formatterの出力（*.txt）
│   │   └── word_embeding/
│   │       ├── Word2Vec/*.libsvm        # Word2Vec処理済み
│   │       ├── FastText/*.libsvm        # FastText処理済み
│   │       └── BERT/*.libsvm            # BERT処理済み
│   └── models/
│       ├── Word2Vec/
│       │   ├── best_model_word2vec.joblib
│       │   ├── training_results_word2vec.txt
│       │   └── model_comparison_word2vec.png 等
│       └── ...
```

---

## 🧩 各ファイルの役割（まとめ）

| ファイル                        | 役割                                 |
| --------------------------- | ---------------------------------- |
| `log_formatter.py`          | 生ログをプレイヤー別のラベル付きテキストに整形            |
| `bert_processor.py`         | BERTで発話を埋め込み、libsvm形式で保存           |
| `fasttext_processor.py`     | FastTextで発話を埋め込み、libsvm形式で保存       |
| `word2vec_processor.py`     | Word2Vecで発話を埋め込み、libsvm形式で保存       |
| `train_werewolf_models.py`  | 高精度モデル（GridSearch付き）を訓練 + 評価 + 可視化 |
| `train_remaining_models.py` | 軽量モデル（簡易設定）を訓練                     |

---

## 🔖 備考

* MeCabやtransformers、fasttextなどの外部ライブラリが存在しない環境でも代替処理（TF-IDF+SVD）で動作可能。
* 各処理は例外処理を含み、欠損ファイルや構文エラーに対して堅牢に対応。

---
