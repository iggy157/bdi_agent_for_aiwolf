#!/usr/bin/env python3
"""
Word2Vecを使用したワードエンベディング処理スクリプト

/judgement_werewolf/libsvm/datasets内のtxtファイルを読み込み、
Word2Vecでエンベディングし、libsvm形式で保存する
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
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not available. Using TF-IDF based Word2Vec alternative.")
    GENSIM_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not available. Using simple tokenization.")
    MECAB_AVAILABLE = False


class Word2VecProcessor:
    """Word2Vec処理クラス"""
    
    def __init__(self, 
                 source_path: str = "judgement_werewolf/libsvm/datasets/data",
                 output_path: str = "judgement_werewolf/libsvm/datasets/word_embeding/Word2Vec",
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 1,
                 epochs: int = 10):
        
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        
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
    
    def process_with_gensim(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """Gensimを使用したWord2Vec処理"""
        print("Processing with Gensim Word2Vec...")
        
        # 全テキストを収集してモデル訓練用のコーパスを作成
        all_sentences = []
        for filename, data_pairs in data_files:
            for label, text in data_pairs:
                tokens = self.tokenize_japanese(text)
                if tokens:
                    all_sentences.append(tokens)
        
        if not all_sentences:
            print("No sentences found for Word2Vec training")
            return
        
        print(f"Training Word2Vec model on {len(all_sentences)} sentences...")
        
        # Word2Vecモデルの訓練
        model = Word2Vec(
            sentences=all_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs
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
                        if token in model.wv:
                            vectors.append(model.wv[token])
                    
                    if vectors:
                        avg_vector = np.mean(vectors, axis=0)
                        embeddings.append(avg_vector)
                        labels.append(label)
            
            if embeddings:
                output_file = self.output_path / f"{filename.replace('.txt', '.libsvm')}"
                self.save_libsvm_format(embeddings, labels, output_file)
    
    def process_with_tfidf_alternative(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """TF-IDFベースのWord2Vec代替処理"""
        print("Processing with TF-IDF based Word2Vec alternative...")
        
        # 全テキストを収集
        all_texts = []
        for filename, data_pairs in data_files:
            for label, text in data_pairs:
                tokens = self.tokenize_japanese(text)
                if tokens:
                    all_texts.append(" ".join(tokens))
        
        if not all_texts:
            print("No texts found for processing")
            return
        
        # TF-IDFベクトライザーで単語レベル特徴抽出
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 1),  # 単語レベル
            min_df=2,
            max_df=0.8
        )
        
        # 全データでフィット
        vectorizer.fit(all_texts)
        
        # SVDで次元削減（Word2Vecっぽく指定次元に）
        svd = TruncatedSVD(n_components=self.vector_size, random_state=42)
        
        # 各ファイルを処理
        for filename, data_pairs in data_files:
            embeddings = []
            labels = []
            texts = []
            
            for label, text in data_pairs:
                tokens = self.tokenize_japanese(text)
                if tokens:
                    texts.append(" ".join(tokens))
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
        
        if GENSIM_AVAILABLE:
            self.process_with_gensim(data_files)
        else:
            self.process_with_tfidf_alternative(data_files)
        
        print("Word2Vec processing completed!")


def main():
    """メイン実行関数"""
    processor = Word2VecProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()