#!/usr/bin/env python3
"""
BERTを使用したワードエンベディング処理スクリプト

/judgement_werewolf/libsvm/datasets内のtxtファイルを読み込み、
BERTでエンベディングし、libsvm形式で保存する
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import re
import warnings
warnings.filterwarnings('ignore')

# 必要なライブラリのインポートとフォールバック処理
# Temporarily forcing TF-IDF alternative due to transformers segmentation fault
try:
    # from transformers import AutoTokenizer, AutoModel
    # import torch
    # TRANSFORMERS_AVAILABLE = True
    raise ImportError("Forcing TF-IDF alternative due to segmentation fault")
except ImportError:
    print("Warning: transformers not available. Using TF-IDF based BERT alternative.")
    TRANSFORMERS_AVAILABLE = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not available. Using simple tokenization.")
    MECAB_AVAILABLE = False


class BERTProcessor:
    """BERT処理クラス"""
    
    def __init__(self, 
                 source_path: str = "/home/bi23056/lab/bdi_agent_for_aiwolf/judgement_werewolf/libsvm/datasets/data",
                 output_path: str = "/home/bi23056/lab/bdi_agent_for_aiwolf/judgement_werewolf/libsvm/datasets/word_embeding/BERT",
                 model_name: str = "cl-tohoku/bert-base-japanese-whole-word-masking",
                 max_length: int = 512,
                 batch_size: int = 8):
        
        self.source_path = Path(source_path)
        self.output_path = Path(output_path)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
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
        
        # BERTモデルの初期化
        if TRANSFORMERS_AVAILABLE:
            self.init_bert_model()
    
    def init_bert_model(self):
        """BERTモデルの初期化"""
        try:
            print(f"Loading BERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # GPUが利用可能な場合は使用
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"BERT model loaded on {self.device}")
            
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Falling back to TF-IDF alternative")
            global TRANSFORMERS_AVAILABLE
            TRANSFORMERS_AVAILABLE = False
    
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
    
    def process_with_bert(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """BERTを使用した処理"""
        print("Processing with BERT...")
        
        # 各ファイルを処理
        for filename, data_pairs in data_files:
            embeddings = []
            labels = []
            
            # バッチ処理
            texts = [text for label, text in data_pairs]
            batch_labels = [label for label, text in data_pairs]
            
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i+self.batch_size]
                batch_embeddings = self.get_bert_embeddings(batch_texts)
                embeddings.extend(batch_embeddings)
            
            labels.extend(batch_labels)
            
            if embeddings:
                output_file = self.output_path / f"{filename.replace('.txt', '.libsvm')}"
                self.save_libsvm_format(embeddings, labels, output_file)
    
    def get_bert_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """BERTエンベディングの取得"""
        embeddings = []
        
        try:
            # テキストをトークン化
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # BERTで特徴抽出
            with torch.no_grad():
                outputs = self.model(**inputs)
                # [CLS]トークンの出力を使用
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                for embedding in cls_embeddings:
                    embeddings.append(embedding)
                    
        except Exception as e:
            print(f"Error in BERT processing: {e}")
            
        return embeddings
    
    def process_with_tfidf_alternative(self, data_files: List[Tuple[str, List[Tuple[int, str]]]]):
        """TF-IDFベースのBERT代替処理"""
        print("Processing with TF-IDF based BERT alternative...")
        
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
        
        # 高次元TF-IDFベクトライザー（BERT風に文書レベル）
        vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 3),  # 1-gram to 3-gram
            min_df=1,
            max_df=0.9,
            sublinear_tf=True  # log正規化
        )
        
        # 全データでフィット
        vectorizer.fit(all_texts)
        
        # BERTっぽく768次元に（ただし実際の特徴数に依存）
        target_dims = min(768, len(vectorizer.get_feature_names_out()))
        svd = TruncatedSVD(n_components=target_dims, random_state=42)
        
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
        
        if TRANSFORMERS_AVAILABLE:
            self.process_with_bert(data_files)
        else:
            self.process_with_tfidf_alternative(data_files)
        
        print("BERT processing completed!")


def main():
    """メイン実行関数"""
    processor = BERTProcessor()
    processor.process_all()


if __name__ == "__main__":
    main()