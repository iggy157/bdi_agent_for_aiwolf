# ワードエンベディング処理スクリプト

人狼ゲームのテキストデータをワードエンベディングしてlibsvm形式に変換するスクリプト集

## 概要

`/judgement_werewolf/libsvm/datasets/`内のtxtファイル（形式: `ラベル,テキスト`）を読み込み、3つの異なるワードエンベディング手法で処理し、libsvm形式で保存します。

## ファイル構成

- `word2vec_processor.py` - Word2Vecによるエンベディング処理
- `fasttext_processor.py` - FastTextによるエンベディング処理  
- `bert_processor.py` - BERTによるエンベディング処理
- `run_all_embeddings.py` - 全手法を統合実行するスクリプト
- `README.md` - このファイル

## 使用方法

### 個別実行

```bash
# Word2Vec処理のみ
python word2vec_processor.py

# FastText処理のみ
python fasttext_processor.py

# BERT処理のみ
python bert_processor.py
```

### 全手法一括実行

```bash
# 3つの手法すべてを実行
python run_all_embeddings.py
```

## 出力

各手法で処理されたファイルは以下のディレクトリに保存されます：

- `Word2Vec/` - Word2Vec処理結果（100次元）
- `FastText/` - FastText処理結果（100次元）
- `BERT/` - BERT処理結果（768次元 or 代替手法の次元数）

## 出力形式

libsvm形式で保存されます：

```
-1 1:0.123456 3:0.789012 5:0.345678
1 2:0.456789 4:0.123456 6:0.789012
```

- 最初の数値: ラベル（1=人狼、-1=人間）
- 後続: `特徴番号:値` の形式

## 依存ライブラリ

### 推奨ライブラリ

```bash
pip install gensim fasttext transformers torch scikit-learn numpy
```

### フォールバック

必要なライブラリがインストールされていない場合、scikit-learnベースの代替手法で動作します：

- **Word2Vec代替**: TF-IDF + SVD
- **FastText代替**: 文字n-gram TF-IDF + SVD  
- **BERT代替**: 高次元TF-IDF + SVD

### 日本語処理（オプション）

```bash
pip install mecab-python3
```

MeCabがない場合は正規表現ベースの簡易トークナイザーを使用します。

## 処理詳細

### Word2Vec処理
- 全テキストでWord2Vecモデルを訓練
- 各文書の単語ベクトルを平均化
- 100次元のベクトル表現を生成

### FastText処理
- サブワード情報を考慮したエンベディング
- 文字n-gramによる未知語対応
- 100次元のベクトル表現を生成

### BERT処理
- 事前訓練済み日本語BERTモデルを使用
- [CLS]トークンの表現を文書ベクトルとして使用
- 768次元のベクトル表現を生成

## 設定のカスタマイズ

各プロセッサのコンストラクタでパラメータを調整可能：

```python
# Word2Vec設定例
processor = Word2VecProcessor(
    vector_size=200,  # ベクトル次元数
    window=10,        # 文脈ウィンドウサイズ
    epochs=20         # 学習エポック数
)

# FastText設定例
processor = FastTextProcessor(
    dim=200,          # ベクトル次元数
    epoch=20,         # 学習エポック数
    min_count=2       # 最小出現回数
)

# BERT設定例
processor = BERTProcessor(
    model_name="cl-tohoku/bert-base-japanese",  # 使用モデル
    max_length=256,                             # 最大トークン長
    batch_size=16                               # バッチサイズ
)
```

## 注意事項

1. **メモリ使用量**: BERTは大量のメモリを使用します
2. **処理時間**: BERTは特に時間がかかります（GPU推奨）
3. **ファイル数**: 254個のファイル × 3手法 = 762個のlibsvmファイルが生成されます

## トラブルシューティング

### CUDA out of memory
BERTでGPUメモリ不足の場合：
```python
processor = BERTProcessor(batch_size=1)  # バッチサイズを小さく
```

### ライブラリ不足
代替手法が自動的に使用されますが、元のライブラリのインストールを推奨：
```bash
pip install gensim fasttext transformers
```