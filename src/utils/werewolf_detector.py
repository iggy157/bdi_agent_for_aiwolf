#!/usr/bin/env python3
"""
人狼判定処理モジュール

libsvm.ymlファイルのデータが4行以上になったとき、
機械学習モデルを使用して人狼判定を行い、
status.ymlファイルに結果を反映する
"""

import os
import yaml
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class WerewolfDetector:
    """人狼判定処理クラス"""
    
    def __init__(self, 
                 model_path: str = "/home/bi23056/lab/aiwolf-nlp-agent-llm/judgement_werewolf/libsvm/models/Word2Vec/best_model_word2vec.joblib",
                 base_libsvm_path: str = "/info/libsvm",
                 base_status_path: str = "/info/status",
                 min_data_threshold: int = 4):
        
        self.model_path = Path(model_path)
        self.base_libsvm_path = Path(base_libsvm_path)
        self.base_status_path = Path(base_status_path)
        self.min_data_threshold = min_data_threshold
        
        # モデルの読み込み
        self.model = None
        self.load_model()
        
        # 処理済みデータ数の追跡（game/agent/from -> 処理済み行数）
        self.processed_counts = {}
    
    def load_model(self):
        """機械学習モデルの読み込み"""
        try:
            if self.model_path.exists():
                # 複数のパターンでモデル読み込みを試行
                try:
                    self.model = joblib.load(self.model_path)
                    print(f"Loaded werewolf detection model: {self.model_path}")
                except Exception as e1:
                    # pickle で直接読み込みを試行
                    try:
                        import pickle
                        with open(self.model_path, 'rb') as f:
                            self.model = pickle.load(f)
                        print(f"Loaded model using pickle: {self.model_path}")
                    except Exception as e2:
                        print(f"Error loading model with joblib: {e1}")
                        print(f"Error loading model with pickle: {e2}")
                        self.model = None
            else:
                print(f"Warning: Model file not found: {self.model_path}")
        except Exception as e:
            print(f"Error in load_model: {e}")
            self.model = None
    
    def parse_libsvm_line(self, line: str) -> Optional[np.ndarray]:
        """libsvm形式の行をパースして特徴ベクトルを取得"""
        try:
            line = line.strip()
            if not line:
                return None
            
            parts = line.split()
            if len(parts) < 2:
                return None
            
            # ラベル部分をスキップ（最初の部分）
            features = {}
            for part in parts[1:]:
                if ':' in part:
                    index, value = part.split(':', 1)
                    features[int(index)] = float(value)
            
            if not features:
                return None
            
            # 特徴ベクトルを作成（インデックス1-100を想定）
            max_index = max(features.keys()) if features else 100
            vector_size = max(100, max_index)  # 最低100次元
            
            vector = np.zeros(vector_size)
            for index, value in features.items():
                if 1 <= index <= vector_size:
                    vector[index - 1] = value  # インデックスは1始まりなので-1
            
            return vector
            
        except Exception as e:
            print(f"Error parsing libsvm line '{line}': {e}")
            return None
    
    def load_libsvm_data(self, libsvm_path: Path) -> List[np.ndarray]:
        """libsvm.ymlファイルからデータを読み込み"""
        try:
            if not libsvm_path.exists():
                return []
            
            vectors = []
            with open(libsvm_path, 'r', encoding='utf-8') as f:
                for line in f:
                    vector = self.parse_libsvm_line(line)
                    if vector is not None:
                        vectors.append(vector)
            
            return vectors
            
        except Exception as e:
            print(f"Error loading libsvm data from {libsvm_path}: {e}")
            return []
    
    def predict_werewolf(self, vectors: List[np.ndarray]) -> List[bool]:
        """ベクトルから人狼判定を実行"""
        if not vectors:
            return []
        
        # モデルが利用できない場合は簡易的な判定を行う
        if not self.model:
            print("Warning: No model available, using fallback prediction")
            # 簡易的な判定：ベクトルの平均値に基づく
            werewolf_predictions = []
            for vector in vectors:
                # ベクトルの平均値がしきい値を超えるかで判定
                mean_value = np.mean(vector)
                is_werewolf = mean_value > 0.005  # 調整可能なしきい値
                werewolf_predictions.append(is_werewolf)
            return werewolf_predictions
        
        try:
            # numpy配列に変換
            X = np.array(vectors)
            
            # 予測実行
            predictions = self.model.predict(X)
            
            # ラベルをbooleanに変換（1: werewolf=True, -1: werewolf=False）
            werewolf_predictions = [pred == 1 for pred in predictions]
            
            return werewolf_predictions
            
        except Exception as e:
            print(f"Error in werewolf prediction: {e}")
            # フォールバック予測
            werewolf_predictions = []
            for vector in vectors:
                mean_value = np.mean(vector)
                is_werewolf = mean_value > 0.005
                werewolf_predictions.append(is_werewolf)
            return werewolf_predictions
    
    def load_status_file(self, status_path: Path) -> Optional[Dict]:
        """status.ymlファイルの読み込み"""
        try:
            if not status_path.exists():
                return None
            
            with open(status_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    return None
                return yaml.safe_load(content)
                
        except Exception as e:
            print(f"Error loading status file {status_path}: {e}")
            return None
    
    def save_status_file(self, status_data: Dict, status_path: Path):
        """status.ymlファイルの保存"""
        try:
            status_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(status_path, 'w', encoding='utf-8') as f:
                yaml.dump(status_data, f, allow_unicode=True, default_flow_style=False)
            
        except Exception as e:
            print(f"Error saving status file {status_path}: {e}")
    
    def get_latest_turn(self, status_data: Dict) -> Optional[int]:
        """status.ymlから最新のターン番号を取得"""
        try:
            turn_numbers = []
            for key in status_data.keys():
                if isinstance(key, int):
                    turn_numbers.append(key)
                elif isinstance(key, str) and key.isdigit():
                    turn_numbers.append(int(key))
            
            return max(turn_numbers) if turn_numbers else None
            
        except Exception as e:
            print(f"Error getting latest turn: {e}")
            return None
    
    def update_status_with_werewolf_prediction(self, game_name: str, agent_name: str, 
                                             from_agent: str, is_werewolf: bool):
        """status.ymlに人狼判定結果を反映"""
        status_path = self.base_status_path / game_name / agent_name / "status.yml"
        
        # status.ymlを読み込み
        status_data = self.load_status_file(status_path)
        if not status_data:
            print(f"No status data found for {game_name}/{agent_name}")
            return
        
        # 最新ターンを取得
        latest_turn = self.get_latest_turn(status_data)
        if latest_turn is None:
            print(f"No valid turn found in status data for {game_name}/{agent_name}")
            return
        
        # 最新ターンのデータを取得
        turn_data = status_data.get(latest_turn, {})
        
        # from_agentに対応するエージェントデータを探す
        agent_key = f"agent    {from_agent}"
        if agent_key in turn_data:
            # 現在のwerewolf値をチェック
            current_werewolf = turn_data[agent_key].get('werewolf', False)
            
            # falseからtrueの場合のみ更新（一度trueになったらfalseに戻さない）
            if not current_werewolf and bool(is_werewolf):
                turn_data[agent_key]['werewolf'] = True
                status_data[latest_turn] = turn_data
                
                # ファイルに保存
                self.save_status_file(status_data, status_path)
                
                print(f"Updated {game_name}/{agent_name}: {from_agent} -> werewolf: False to True")
            elif not current_werewolf and not bool(is_werewolf):
                # falseのままの場合は明示的に設定（まだwerewolfフィールドがない場合用）
                if 'werewolf' not in turn_data[agent_key]:
                    turn_data[agent_key]['werewolf'] = False
                    status_data[latest_turn] = turn_data
                    self.save_status_file(status_data, status_path)
                    print(f"Initialized {game_name}/{agent_name}: {from_agent} -> werewolf: False")
            else:
                print(f"No update needed for {game_name}/{agent_name}: {from_agent} (current: {current_werewolf}, predicted: {bool(is_werewolf)})")
        else:
            print(f"Agent '{from_agent}' not found in turn {latest_turn} for {game_name}/{agent_name}")
    
    def process_libsvm_file(self, game_name: str, agent_name: str, from_agent: str):
        """単一のlibsvm.ymlファイルを処理"""
        libsvm_path = self.base_libsvm_path / game_name / agent_name / from_agent / "libsvm.yml"
        
        # データを読み込み
        vectors = self.load_libsvm_data(libsvm_path)
        
        if len(vectors) < self.min_data_threshold:
            return  # データ不足
        
        # 処理済み件数を確認
        key = f"{game_name}/{agent_name}/{from_agent}"
        processed_count = self.processed_counts.get(key, 0)
        
        if len(vectors) <= processed_count:
            return  # 新しいデータなし
        
        # 新しいデータのみ処理
        new_vectors = vectors[processed_count:]
        
        if not new_vectors:
            return
        
        # 人狼判定を実行
        predictions = self.predict_werewolf(new_vectors)
        
        if not predictions:
            return
        
        # 最新の判定結果を取得（複数ある場合は最後の結果を使用）
        latest_prediction = predictions[-1]
        
        # status.ymlを更新
        self.update_status_with_werewolf_prediction(
            game_name, agent_name, from_agent, latest_prediction
        )
        
        # 処理済み件数を更新
        self.processed_counts[key] = len(vectors)
        
        print(f"Processed werewolf detection for {key}: {latest_prediction} (new data: {len(new_vectors)})")
    
    def scan_and_process_all(self):
        """全てのlibsvm.ymlファイルをスキャンして処理"""
        if not self.base_libsvm_path.exists():
            print(f"libsvm base path does not exist: {self.base_libsvm_path}")
            return
        
        # libsvm.ymlファイルを検索
        libsvm_files = list(self.base_libsvm_path.glob("*/*/*/libsvm.yml"))
        
        for libsvm_file in libsvm_files:
            try:
                # パスから情報を抽出
                parts = libsvm_file.parts
                libsvm_idx = -1
                for i, part in enumerate(parts):
                    if part == 'libsvm':
                        libsvm_idx = i
                        break
                
                if libsvm_idx == -1 or libsvm_idx + 3 >= len(parts):
                    continue
                
                game_name = parts[libsvm_idx + 1]
                agent_name = parts[libsvm_idx + 2]
                from_agent = parts[libsvm_idx + 3]
                
                self.process_libsvm_file(game_name, agent_name, from_agent)
                
            except Exception as e:
                print(f"Error processing {libsvm_file}: {e}")


def main():
    """メイン実行関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Werewolf detection from libsvm data')
    parser.add_argument('--model-path', 
                       default='/home/bi23056/lab/aiwolf-nlp-agent-llm/judgement_werewolf/libsvm/models/Word2Vec/best_model_word2vec.joblib',
                       help='Path to the trained model file')
    parser.add_argument('--libsvm-path', default='/info/libsvm',
                       help='Base path for libsvm files')
    parser.add_argument('--status-path', default='/info/status',
                       help='Base path for status files')
    parser.add_argument('--min-threshold', type=int, default=4,
                       help='Minimum data threshold for detection')
    parser.add_argument('--continuous', action='store_true',
                       help='Run continuously (for integration with file watcher)')
    
    args = parser.parse_args()
    
    # 検出器を初期化
    detector = WerewolfDetector(
        model_path=args.model_path,
        base_libsvm_path=args.libsvm_path,
        base_status_path=args.status_path,
        min_data_threshold=args.min_threshold
    )
    
    if args.continuous:
        print("Running in continuous mode...")
        import time
        try:
            while True:
                detector.scan_and_process_all()
                time.sleep(5)  # 5秒間隔でチェック
        except KeyboardInterrupt:
            print("\nStopping werewolf detector...")
    else:
        print("Processing all existing libsvm files...")
        detector.scan_and_process_all()
        print("Werewolf detection completed!")


if __name__ == "__main__":
    main()