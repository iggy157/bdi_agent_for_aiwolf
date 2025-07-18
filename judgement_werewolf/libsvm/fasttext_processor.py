#!/usr/bin/env python3
"""
FastTextを使用したワードエンベディング処理スクリプト

/judgement_werewolf/libsvm/datasets内のtxtファイルを読み込み、
FastTextでエンベディングし、libsvm形式で保存する
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

# 必要なライブラリのインポートとフォールバック処理
try:
    import fasttext
    FASTTEXT_AVAILABLE = True
except ImportError:
    print("Warning: fasttext not available. Using character n-gram based alternative.")
    FASTTEXT_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not available. Using simple tokenization.")
    MECAB_AVAILABLE = False


class FastTextProcessor:
    """FastText処理クラス"""
    
    def __init__(self, 
                 source_path: str = "/home/bi23056/lab/bdi_agent_for_aiwolf/judgement_werewolf/libsvm/datasets/data",
                 output_path: str = "/home/bi23056/lab/bdi_agent_for_aiwolf/judgement_werewolf/libsvm/datasets/word_embeding/FastText",
                 dim: int = 100,
                 epoch: int = 10,
                 min_count: int = 1):
        
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.dim = dim
        self.epoch = epoch
        self.min_count = min_count
        
        # 出力ディレクトリを作成
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 日本語形態素解析器の初期化
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger("-Owakati")
            except:
                self.mecab = None
                print("Warning: MeCab initialization failed")
        else:
            self.mecab = None
    
    def tokenize_japanese(self, text: str) -> List[str]:
        """日本語テキストの分かち書き"""
        if self.mecab:
            try:
                tokens = self.mecab.parse(text).strip().split()
                return [token for token in tokens if token and len(token) > 0]
            except:
                pass
        
        # MeCabが使えない場合は文字単位で分割
        japanese_pattern = re.compile(r'[ひらがなカタカナ漢字a-zA-Z0-9]+')
        tokens = japanese_pattern.findall(text)
        return tokens
    
    def extract_char_ngrams(self, text: str, n: int = 2) -> List[str]:
        """文字n-gramの抽出（FastTextスタイル）"""
        tokens = self.tokenize_japanese(text)
        ngrams = []
        for token in tokens:
            if len(token) >= n:
                for i in range(len(token) - n + 1):
                    ngrams.append(token[i:i+n])
        return ngrams
    
    def load_data_files(self) -> List[Tuple[str, List[Tuple[int, str]]]]:
        """データファイルの読み込み"""
        data_files = []
        txt_files = list(self.source_path.glob("*.txt"))
        
        for txt_file in txt_files:
            try:
                data_pairs = []
                with open(txt_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and ',' in line:
                            parts = line.split(',', 1)
                            if len(parts) == 2:
                                try:
                                    label = int(parts[0])
                                    text = parts[1]
                                    data_pairs.append((label, text))
                                except ValueError:
                                    continue
                
                if data_pairs:
                    data_files.append((txt_file.name, data_pairs))
                    
            except Exception as e:
                print(f"Error loading {txt_file.name}: {e}")
        
        return data_files
    
    def process_with_fasttext(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """FastTextを使用した処理"""
        print("Processing with FastText...")
        
        # FastText訓練用の一時ファイルを作成
        temp_file = self.output_path.parent / "temp_fasttext_corpus.txt"
        
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for filename, data_pairs in data_files:
                    for label, text in data_pairs:
                        tokens = self.tokenize_japanese(text)
                        if tokens:
                            f.write(" ".join(tokens) + "\n")
            
            print(f"Training FastText model...")
            
            # FastTextモデルの訓練
            model = fasttext.train_unsupervised(
                str(temp_file),
                model='skipgram',
                dim=self.dim,
                epoch=self.epoch,
                minCount=self.min_count
            )
            
            # 各ファイルを処理
            for filename, data_pairs in data_files:
                embeddings = []
                labels = []
                
                for label, text in data_pairs:
                    tokens = self.tokenize_japanese(text)
                    if tokens:
                        # 単語ベクトルの平均を計算
                        vectors = []
                        for token in tokens:
                            vectors.append(model.get_word_vector(token))
                        
                        if vectors:
                            avg_vector = np.mean(vectors, axis=0)
                            embeddings.append(avg_vector)
                            labels.append(label)
                
                if embeddings:
                    output_file = self.output_path / f"{filename.replace('.txt', '.libsvm')}"
                    self.save_libsvm_format(embeddings, labels, output_file)
            
            # 一時ファイルを削除
            temp_file.unlink()
            
        except Exception as e:
            print(f"Error in FastText processing: {e}")
            if temp_file.exists():
                temp_file.unlink()
    
    def process_with_ngram_alternative(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """文字n-gramベースのFastText代替処理"""
        print("Processing with character n-gram based FastText alternative...")
        
        # 全テキストを収集して文字n-gramに変換
        all_texts = []
        for filename, data_pairs in data_files:
            for label, text in data_pairs:
                ngrams = self.extract_char_ngrams(text, n=2)
                if ngrams:
                    all_texts.append(" ".join(ngrams))
        
        if not all_texts:
            print("No texts found for processing")
            return
        
        # 文字n-gramレベルのTF-IDFベクトライザー
        vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),  # 1-gram and 2-gram of character n-grams
            min_df=2,
            max_df=0.8
        )
        
        # 全データでフィット
        vectorizer.fit(all_texts)
        
        # SVDで次元削減
        svd = TruncatedSVD(n_components=self.dim, random_state=42)
        
        # 各ファイルを処理
        for filename, data_pairs in data_files:
            embeddings = []
            labels = []
            texts = []
            
            for label, text in data_pairs:
                ngrams = self.extract_char_ngrams(text, n=2)
                if ngrams:
                    texts.append(" ".join(ngrams))
                    labels.append(label)
            
            if texts:
                # TF-IDF変換
                tfidf_matrix = vectorizer.transform(texts)
                
                # SVDで次元削減
                if tfidf_matrix.shape[0] > 0:
                    reduced_features = svd.fit_transform(tfidf_matrix)
                    
                    for reduced_vector in reduced_features:
                        embeddings.append(reduced_vector)
                    
                    output_file = self.output_path / f"{filename.replace('.txt', '.libsvm')}"
                    self.save_libsvm_format(embeddings, labels, output_file)
    
    def save_libsvm_format(self, embeddings: List[np.ndarray], labels: List[int], output_path: Path):
        """libsvm形式でファイルを保存"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for embedding, label in zip(embeddings, labels):
                    # ラベルを人狼判定用に変換 (1: 人狼, -1: 人間)
                    libsvm_label = 1 if label == 1 else -1
                    
                    # 特徴量を文字列化
                    features = []
                    for i, value in enumerate(embedding, 1):
                        if abs(value) > 1e-6:  # 非常に小さい値は0とみなす
                            features.append(f"{i}:{value:.6f}")
                    
                    if features:  # 特徴量がある場合のみ保存
                        line = f"{libsvm_label} " + " ".join(features)
                        f.write(line + "\n")
            
            print(f"Saved: {output_path.name} ({len(embeddings)} samples)")
            
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
    
    def process_all(self):
        """全処理の実行"""
        print("Loading data files...")
        data_files = self.load_data_files()
        
        if not data_files:
            print("No data files found")
            return
        
        print(f"Found {len(data_files)} data files")
        
        if FASTTEXT_AVAILABLE:
            self.process_with_fasttext(data_files)
        else:
            self.process_with_ngram_alternative(data_files)
        
        print("FastText processing completed!")


def main():
    """メイン実行関数"""
    processor = FastTextProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()