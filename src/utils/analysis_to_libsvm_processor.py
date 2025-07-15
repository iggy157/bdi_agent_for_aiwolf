#!/usr/bin/env python3
"""
analysis.yml ファイルから libsvm 形式への変換処理

/info/analysis/{game_name}/{agent_name}/analysis.yml のデータを
エージェントの from ごとに分類し、Word2Vec で libsvm 形式に変換して
/info/libsvm/{game_name}/{agent_name}/{from}/libsvm.yml に保存する
"""

import os
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# watchdog のインポート（オプショナル）
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("Warning: watchdog not available. File watching disabled.")
    WATCHDOG_AVAILABLE = False
    FileSystemEventHandler = object  # ダミークラス
    FileModifiedEvent = None

# 必要なライブラリのインポートとフォールバック処理
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not available. Using TF-IDF based Word2Vec alternative.")
    GENSIM_AVAILABLE = False

# sklearn のインポート（オプショナル）
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: sklearn not available.")
    SKLEARN_AVAILABLE = False
    TfidfVectorizer = None
    TruncatedSVD = None

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not available. Using simple tokenization.")
    MECAB_AVAILABLE = False


class AnalysisToLibsvmProcessor:
    """analysis.yml を libsvm 形式に変換するプロセッサ"""
    
    def __init__(self, 
                 base_analysis_path: str = "/info/analysis",
                 base_libsvm_path: str = "/info/libsvm",
                 vector_size: int = 100,
                 window: int = 5,
                 min_count: int = 1,
                 epochs: int = 10):
        
        self.base_analysis_path = Path(base_analysis_path)
        self.base_libsvm_path = Path(base_libsvm_path)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        
        # 処理済みエントリのトラッキング（ゲーム/エージェント/送信元ごと）
        self.processed_entries = {}  # {game_name: {agent_name: {from: set(entry_keys)}}}
        
        # 日本語形態素解析器の初期化
        if MECAB_AVAILABLE:
            try:
                self.mecab = MeCab.Tagger("-Owakati")
            except:
                self.mecab = None
                print("Warning: MeCab initialization failed")
        else:
            self.mecab = None
            
        # Word2Vecモデルのキャッシュ（ゲームごと）
        self.model_cache = {}
    
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
    
    def load_analysis_file(self, file_path: Path) -> Optional[Dict]:
        """analysis.yml ファイルの読み込み"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return None
                    
                # YAMLの各エントリを個別に読み込む
                entries = {}
                current_entry = []
                
                for line in content.split('\n'):
                    if line.strip() and not line.startswith('#'):
                        if ':' in line and not line.startswith('  '):
                            # 新しいエントリの開始
                            if current_entry:
                                # 前のエントリを処理
                                entry_text = '\n'.join(current_entry)
                                try:
                                    entry_data = yaml.safe_load(entry_text)
                                    if entry_data:
                                        entries.update(entry_data)
                                except:
                                    pass
                            current_entry = [line]
                        else:
                            current_entry.append(line)
                
                # 最後のエントリを処理
                if current_entry:
                    entry_text = '\n'.join(current_entry)
                    try:
                        entry_data = yaml.safe_load(entry_text)
                        if entry_data:
                            entries.update(entry_data)
                    except:
                        pass
                
                return entries
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def group_by_sender(self, analysis_data: Dict) -> Dict[str, List[Tuple[str, Dict]]]:
        """analysis データを送信元（from）ごとにグループ化"""
        grouped_data = {}
        
        for key, entry in analysis_data.items():
            if isinstance(entry, dict) and 'from' in entry and 'content' in entry:
                sender = entry['from']
                if sender not in grouped_data:
                    grouped_data[sender] = []
                grouped_data[sender].append((entry['content'], entry))
        
        return grouped_data
    
    def train_word2vec_model(self, all_texts: List[str]) -> Optional[Word2Vec]:
        """Word2Vec モデルの訓練"""
        if not GENSIM_AVAILABLE:
            return None
            
        # テキストをトークン化
        all_sentences = []
        for text in all_texts:
            tokens = self.tokenize_japanese(text)
            if tokens:
                all_sentences.append(tokens)
        
        if not all_sentences:
            return None
        
        # Word2Vec モデルの訓練
        model = Word2Vec(
            sentences=all_sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=4,
            epochs=self.epochs
        )
        
        return model
    
    def text_to_vector(self, text: str, model: Optional[Word2Vec] = None, 
                      vectorizer: Optional[object] = None, 
                      svd: Optional[object] = None) -> Optional[np.ndarray]:
        """テキストをベクトルに変換"""
        tokens = self.tokenize_japanese(text)
        if not tokens:
            return None
        
        if model and GENSIM_AVAILABLE:
            # Word2Vec を使用
            vectors = []
            for token in tokens:
                if token in model.wv:
                    vectors.append(model.wv[token])
            
            if vectors:
                return np.mean(vectors, axis=0)
        
        elif vectorizer and svd:
            # TF-IDF + SVD を使用
            text_joined = " ".join(tokens)
            try:
                tfidf_vector = vectorizer.transform([text_joined])
                reduced_vector = svd.transform(tfidf_vector)
                return reduced_vector[0]
            except:
                pass
        
        return None
    
    def save_libsvm_format(self, vectors: List[np.ndarray], metadata: List[Dict], 
                          output_path: Path, append: bool = False):
        """libsvm 形式で保存（標準形式：1行1サンプル）"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append and output_path.exists() else 'w'
            
            with open(output_path, mode, encoding='utf-8') as f:
                for vector, meta in zip(vectors, metadata):
                    # ラベルは0（デフォルト）、必要に応じて分析タイプから決定可能
                    label = 0
                    
                    # 特徴量を libsvm 形式で出力: label index1:value1 index2:value2 ...
                    features = []
                    for i, value in enumerate(vector, 1):
                        if abs(value) > 1e-6:  # 非常に小さい値は除外
                            features.append(f"{i}:{value:.6f}")
                    
                    if features:
                        line = f"{label} " + " ".join(features)
                        f.write(line + "\n")
            
            print(f"{'Appended to' if append else 'Saved'} libsvm data: {output_path} ({len(vectors)} entries)")
            
        except Exception as e:
            print(f"Error saving {output_path}: {e}")
    
    def process_analysis_file(self, analysis_file_path: Path, is_update: bool = False):
        """単一の analysis.yml ファイルを処理（差分更新対応）"""
        # パスから game_name と agent_name を取得
        parts = analysis_file_path.parts
        try:
            analysis_idx = parts.index('analysis')
            game_name = parts[analysis_idx + 1]
            agent_name = parts[analysis_idx + 2]
        except (ValueError, IndexError):
            print(f"Invalid path structure: {analysis_file_path}")
            return
        
        print(f"Processing analysis file for game: {game_name}, agent: {agent_name} (update: {is_update})")
        
        # analysis.yml を読み込む
        analysis_data = self.load_analysis_file(analysis_file_path)
        if not analysis_data:
            print(f"No data found in {analysis_file_path}")
            return
        
        # 処理済みエントリの初期化
        if game_name not in self.processed_entries:
            self.processed_entries[game_name] = {}
        if agent_name not in self.processed_entries[game_name]:
            self.processed_entries[game_name][agent_name] = {}
        
        # 送信元ごとにグループ化
        grouped_data = self.group_by_sender(analysis_data)
        
        if not grouped_data:
            print(f"No valid entries found in {analysis_file_path}")
            return
        
        # 全テキストを収集（モデル訓練用）
        all_texts = []
        for sender_data in grouped_data.values():
            for text, _ in sender_data:
                all_texts.append(text)
        
        # Word2Vec モデルを取得または訓練
        model = None
        vectorizer = None
        svd = None
        
        if GENSIM_AVAILABLE:
            if game_name not in self.model_cache or is_update:
                # 更新時は既存のテキストも含めて再訓練
                if is_update and game_name in self.model_cache:
                    print(f"Updating Word2Vec model for game: {game_name}")
                else:
                    print(f"Training Word2Vec model for game: {game_name}")
                model = self.train_word2vec_model(all_texts)
                self.model_cache[game_name] = model
            else:
                model = self.model_cache[game_name]
        else:
            # TF-IDF + SVD を使用
            print(f"Using TF-IDF + SVD for game: {game_name}")
            vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 1),
                min_df=1,
                max_df=0.8
            )
            
            # テキストを結合
            joined_texts = [" ".join(self.tokenize_japanese(text)) for text in all_texts]
            vectorizer.fit(joined_texts)
            
            svd = TruncatedSVD(n_components=self.vector_size, random_state=42)
            tfidf_matrix = vectorizer.transform(joined_texts)
            svd.fit(tfidf_matrix)
        
        # 各送信元ごとに処理
        for sender, sender_data in grouped_data.items():
            # 処理済みエントリのセットを取得
            if sender not in self.processed_entries[game_name][agent_name]:
                self.processed_entries[game_name][agent_name][sender] = set()
            
            processed_set = self.processed_entries[game_name][agent_name][sender]
            
            # 新規エントリのみ処理
            new_vectors = []
            new_metadata = []
            
            for text, entry in sender_data:
                # エントリのキーを生成（コンテンツとrequest_countの組み合わせ）
                entry_key = f"{entry.get('content', '')}_{entry.get('request_count', 0)}"
                
                if entry_key not in processed_set:
                    vector = self.text_to_vector(text, model, vectorizer, svd)
                    if vector is not None:
                        new_vectors.append(vector)
                        new_metadata.append(entry)
                        processed_set.add(entry_key)
            
            if new_vectors:
                # 出力パス
                output_path = self.base_libsvm_path / game_name / agent_name / sender / "libsvm.yml"
                # 更新時は追記モード
                self.save_libsvm_format(new_vectors, new_metadata, output_path, append=is_update)
    
    def process_all_existing_files(self):
        """既存のすべての analysis.yml ファイルを処理"""
        if not self.base_analysis_path.exists():
            print(f"Analysis base path does not exist: {self.base_analysis_path}")
            return
        
        analysis_files = list(self.base_analysis_path.glob("*/*/analysis.yml"))
        print(f"Found {len(analysis_files)} analysis files to process")
        
        for analysis_file in analysis_files:
            self.process_analysis_file(analysis_file)


class AnalysisFileHandler(FileSystemEventHandler):
    """ファイルシステムイベントハンドラー"""
    
    def __init__(self, processor: AnalysisToLibsvmProcessor):
        self.processor = processor
    
    def on_modified(self, event):
        if isinstance(event, FileModifiedEvent) and event.src_path.endswith('analysis.yml'):
            file_path = Path(event.src_path)
            print(f"Detected modification: {file_path}")
            self.processor.process_analysis_file(file_path, is_update=True)
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('analysis.yml'):
            file_path = Path(event.src_path)
            print(f"Detected new file: {file_path}")
            self.processor.process_analysis_file(file_path, is_update=False)


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert analysis.yml to libsvm format')
    parser.add_argument('--analysis-path', default='/info/analysis', 
                       help='Base path for analysis files')
    parser.add_argument('--libsvm-path', default='/info/libsvm', 
                       help='Base path for libsvm output')
    parser.add_argument('--watch', action='store_true', 
                       help='Watch for file changes')
    parser.add_argument('--vector-size', type=int, default=100,
                       help='Word2Vec vector size')
    
    args = parser.parse_args()
    
    # プロセッサを初期化
    processor = AnalysisToLibsvmProcessor(
        base_analysis_path=args.analysis_path,
        base_libsvm_path=args.libsvm_path,
        vector_size=args.vector_size
    )
    
    # 既存ファイルを処理
    print("Processing existing analysis files...")
    processor.process_all_existing_files()
    
    if args.watch:
        if WATCHDOG_AVAILABLE:
            # ファイル監視モードを開始
            print(f"Starting file watcher on {args.analysis_path}...")
            event_handler = AnalysisFileHandler(processor)
            observer = Observer()
            observer.schedule(event_handler, str(processor.base_analysis_path), recursive=True)
            observer.start()
            
            try:
                print("Watching for changes... Press Ctrl+C to stop.")
                import time
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                observer.stop()
                print("\nStopping file watcher...")
            
            observer.join()
        else:
            print("Warning: watchdog module not available. Cannot watch for file changes.")
    
    print("Processing completed!")


if __name__ == "__main__":
    main()