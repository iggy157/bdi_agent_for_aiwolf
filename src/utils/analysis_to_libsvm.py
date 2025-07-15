#!/usr/bin/env python3
"""
/info/analysis/{game_timestamp}/{agent_name}/analysis.yml を
fromごとにWord2Vec+MeCabでlibsvm変換し
/info/libsvm/{game_timestamp}/{agent_name}/{from}/libsvm.yml に保存する

自動監視機能により、analysis.ymlファイルが更新されるたびに自動処理を実行
"""

import yaml
import numpy as np
from pathlib import Path
import re
import warnings
from collections import defaultdict
import time
import threading
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# 必須: pip install gensim mecab-python3 pyyaml watchdog

try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    print("Warning: gensim not available.")
    GENSIM_AVAILABLE = False

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    print("Warning: MeCab not available.")
    MECAB_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("Warning: watchdog not available. Install with: pip install watchdog")
    WATCHDOG_AVAILABLE = False

class AnalysisFileHandler(FileSystemEventHandler):
    """analysis.ymlファイルの変更を監視するハンドラー"""
    def __init__(self, processor_manager):
        self.processor_manager = processor_manager
        self.last_modified = {}
        super().__init__()
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('analysis.yml'):
            # 重複処理を防ぐため、最後の更新時刻をチェック
            current_time = time.time()
            if event.src_path in self.last_modified:
                if current_time - self.last_modified[event.src_path] < 1:  # 1秒以内の重複は無視
                    return
            
            self.last_modified[event.src_path] = current_time
            
            # パスからgame_timestampとagent_nameを抽出
            path_parts = Path(event.src_path).parts
            if len(path_parts) >= 4 and 'analysis' in path_parts:
                analysis_idx = path_parts.index('analysis')
                if analysis_idx + 2 < len(path_parts):
                    game_timestamp = path_parts[analysis_idx + 1]
                    agent_name = path_parts[analysis_idx + 2]
                    
                    print(f"Detected change in analysis.yml: {game_timestamp}/{agent_name}")
                    self.processor_manager.process_file(event.src_path, agent_name, game_timestamp)

class AnalysisProcessorManager:
    """複数のanalysis.ymlファイルを管理・処理するマネージャー"""
    def __init__(self, base_path: str = "info/analysis", vector_size=100, window=5, min_count=1, epochs=10):
        self.base_path = Path(base_path)
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.processing_lock = threading.Lock()
        
    def process_file(self, analysis_path: str, agent_name: str, game_timestamp: str):
        """analysis.ymlファイルを処理"""
        try:
            with self.processing_lock:
                processor = AnalysisWord2VecLibsvm(
                    analysis_path=analysis_path,
                    agent_name=agent_name,
                    game_timestamp=game_timestamp,
                    vector_size=self.vector_size,
                    window=self.window,
                    min_count=self.min_count,
                    epochs=self.epochs
                )
                processor.run()
        except Exception as e:
            print(f"Error processing {analysis_path}: {e}")
    
    def setup_directory_structure(self):
        """ディレクトリ構造の初期化"""
        if not self.base_path.exists():
            print(f"Creating base directory: {self.base_path}")
            self.base_path.mkdir(parents=True, exist_ok=True)
            
        # 既存のanalysis.ymlファイルを検索して初期処理
        existing_files = list(self.base_path.glob("**/analysis.yml"))
        if existing_files:
            print(f"Found {len(existing_files)} existing analysis.yml files")
            for file_path in existing_files:
                path_parts = file_path.parts
                if len(path_parts) >= 4 and 'analysis' in path_parts:
                    analysis_idx = path_parts.index('analysis')
                    if analysis_idx + 2 < len(path_parts):
                        game_timestamp = path_parts[analysis_idx + 1]
                        agent_name = path_parts[analysis_idx + 2]
                        print(f"Processing existing file: {game_timestamp}/{agent_name}")
                        self.process_file(str(file_path), agent_name, game_timestamp)
    
    def start_monitoring(self):
        """ファイル監視を開始"""
        if not WATCHDOG_AVAILABLE:
            print("Error: watchdog not available. Cannot start monitoring.")
            return
            
        # ディレクトリ構造の初期化
        self.setup_directory_structure()
            
        observer = Observer()
        handler = AnalysisFileHandler(self)
        observer.schedule(handler, str(self.base_path), recursive=True)
        observer.start()
        
        print(f"Started monitoring {self.base_path} for analysis.yml changes...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\nStopping file monitoring...")
        
        observer.join()

class AnalysisWord2VecLibsvm:
    """analysis.yml -> Word2Vec+MeCab -> libsvm形式 出力クラス"""
    def __init__(self, analysis_path: Path, agent_name: str, game_timestamp: str, vector_size=100, window=5, min_count=1, epochs=10):
        self.analysis_path = Path(analysis_path)
        self.agent_name = agent_name
        self.game_timestamp = game_timestamp
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.mecab = MeCab.Tagger("-Owakati") if MECAB_AVAILABLE else None

    def tokenize_japanese(self, text: str):
        """MeCab分かち書き（失敗時は簡易トークン）"""
        if self.mecab:
            try:
                return [t for t in self.mecab.parse(text).strip().split() if t]
            except Exception:
                pass
        # Fallback: 文字列から日本語単語っぽい部分を切り出す
        return re.findall(r'[ぁ-んァ-ン一-龥a-zA-Z0-9]+', text)

    def load_analysis_yml(self):
        """YAMLデータ読込"""
        with open(self.analysis_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def group_by_from(self, analysis_dict):
        """fromごとにcontentをまとめる"""
        grouped = defaultdict(list)
        for v in analysis_dict.values():
            if "from" in v and "content" in v:
                grouped[v["from"]].append(v["content"])
        return grouped

    def train_word2vec(self, all_contents):
        """Word2Vecモデル学習"""
        sentences = [self.tokenize_japanese(c) for c in all_contents if c]
        if not sentences:
            raise ValueError("No content found for Word2Vec")
        model = Word2Vec(sentences, vector_size=self.vector_size, window=self.window,
                         min_count=self.min_count, epochs=self.epochs)
        return model

    def content_to_vector(self, content, model):
        """contentをWord2Vec平均ベクトルに"""
        tokens = self.tokenize_japanese(content)
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(model.vector_size)

    def vector_to_libsvm(self, vec):
        """libsvm形式の1行に変換（ラベル0, index:値 1始まり）"""
        features = [f"{i+1}:{x:.6f}" for i, x in enumerate(vec) if abs(x) > 1e-6]
        return "0 " + " ".join(features)

    def save_libsvm(self, lines, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

    def run(self):
        # 1. analysis.yml読込
        analysis_dict = self.load_analysis_yml()
        # 2. fromごとに分類
        grouped = self.group_by_from(analysis_dict)
        # 3. 全contentでWord2Vecモデル学習
        all_contents = []
        for contents in grouped.values():
            all_contents.extend(contents)
        model = self.train_word2vec(all_contents)
        # 4. fromごとにlibsvmデータ生成・保存
        for from_name, contents in grouped.items():
            libsvm_lines = []
            for content in contents:
                vec = self.content_to_vector(content, model)
                line = self.vector_to_libsvm(vec)
                libsvm_lines.append(line)
            save_path = Path("info") / "libsvm" / self.game_timestamp / self.agent_name / from_name / "libsvm.yml"
            self.save_libsvm(libsvm_lines, save_path)
            print(f"saved: {save_path} ({len(libsvm_lines)} lines)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="analysis.yml -> libsvm変換ツール")
    parser.add_argument("--mode", choices=["single", "monitor"], default="single", 
                       help="実行モード: single=単一ファイル処理, monitor=自動監視")
    parser.add_argument("--analysis-path", help="analysis.ymlのパス (single モード用)")
    parser.add_argument("--agent-name", help="このエージェント名 (single モード用)")
    parser.add_argument("--game-timestamp", help="ゲーム開始時刻タイムスタンプ (single モード用)")
    parser.add_argument("--base-path", default="info/analysis", 
                       help="監視するベースパス (monitor モード用)")
    parser.add_argument("--vector-size", type=int, default=100, help="Word2Vec次元")
    parser.add_argument("--window", type=int, default=5, help="Word2Vec window")
    parser.add_argument("--min-count", type=int, default=1, help="Word2Vec min_count")
    parser.add_argument("--epochs", type=int, default=10, help="Word2Vec epochs")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        # 単一ファイル処理モード
        if not all([args.analysis_path, args.agent_name, args.game_timestamp]):
            print("Error: single モードには --analysis-path, --agent-name, --game-timestamp が必要です")
            exit(1)
        
        processor = AnalysisWord2VecLibsvm(
            analysis_path=args.analysis_path,
            agent_name=args.agent_name,
            game_timestamp=args.game_timestamp,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs
        )
        processor.run()
        
    elif args.mode == "monitor":
        # 自動監視モード
        manager = AnalysisProcessorManager(
            base_path=args.base_path,
            vector_size=args.vector_size,
            window=args.window,
            min_count=args.min_count,
            epochs=args.epochs
        )
        manager.start_monitoring()
