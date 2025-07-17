#!/usr/bin/env python3
"""
人狼判定処理モジュール

libsvmファイルの件数を監視し、4行以上になったときに人狼判定を実行して
status.ymlに結果を反映する処理を行う。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Dict, Any
import joblib
import numpy as np
from datetime import datetime, UTC
from ulid import parse as ulid_parse

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info


class WerewolfJudgment:
    """人狼判定を行うクラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.model = None
        self.model_path = Path("judgement_werewolf/libsvm/models/Word2Vec/best_model_word2vec.joblib")
        
        # 処理済みデータ数を追跡（エージェント別ファイル別）
        # {game_id: {agent_name: {from_name: processed_count}}}
        self.processed_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        
        # モデルの読み込み
        self._load_model()
    
    def _load_model(self) -> None:
        """機械学習モデルを読み込み"""
        try:
            if self.model_path.exists():
                # バージョン互換性の問題を解決するためのワークアラウンド
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    self.model = joblib.load(self.model_path)
                print(f"Loaded werewolf judgment model: {self.model_path}")
            else:
                print(f"Warning: Model file not found: {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            print(f"Model path: {self.model_path}")
            # モデルが読み込めない場合でも継続して動作させる
            self.model = None
    
    def get_game_timestamp(self, game_id: str) -> str:
        """
        ゲームIDからタイムスタンプ文字列を取得
        
        Args:
            game_id: ゲームID
            
        Returns:
            タイムスタンプ文字列
        """
        try:
            ulid_obj = ulid_parse(game_id)
            tz = datetime.now(UTC).astimezone().tzinfo
            return datetime.fromtimestamp(ulid_obj.timestamp().int / 1000, tz=tz).strftime(
                "%Y%m%d%H%M%S%f"
            )[:-3]
        except Exception:
            return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    
    def count_libsvm_lines(self, file_path: Path) -> int:
        """
        libsvmファイルの行数をカウント
        
        Args:
            file_path: libsvmファイルのパス
            
        Returns:
            行数
        """
        try:
            if not file_path.exists():
                return 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
                return len(lines)
        except Exception as e:
            print(f"Error counting lines in {file_path}: {e}")
            return 0
    
    def read_libsvm_data(self, file_path: Path) -> Optional[np.ndarray]:
        """
        libsvmファイルからデータを読み込み
        
        Args:
            file_path: libsvmファイルのパス
            
        Returns:
            特徴量データ（numpy配列）、読み込みに失敗した場合はNone
        """
        try:
            if not file_path.exists():
                return None
            
            features_list = []
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # libsvm形式をパース: "label feature1:value1 feature2:value2 ..."
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    
                    # 特徴量を抽出（100次元と仮定）
                    features = np.zeros(100)
                    for part in parts[1:]:  # 最初はラベルなのでスキップ
                        try:
                            idx_str, value_str = part.split(":")
                            idx = int(idx_str) - 1  # libsvmは1-indexedなので0-indexedに変換
                            value = float(value_str)
                            if 0 <= idx < 100:
                                features[idx] = value
                        except ValueError:
                            continue
                    
                    features_list.append(features)
            
            if features_list:
                return np.array(features_list)
            else:
                return None
                
        except Exception as e:
            print(f"Error reading libsvm data from {file_path}: {e}")
            return None
    
    def predict_werewolf(self, features: np.ndarray) -> bool:
        """
        特徴量から人狼判定を実行
        
        Args:
            features: 特徴量データ
            
        Returns:
            人狼かどうか（True=人狼、False=その他）
        """
        if self.model is None:
            print("Warning: Model not loaded, using fallback prediction")
            # モデルが使えない場合のフォールバック処理
            # 特徴量の平均値を使った簡単なヒューリスティック
            if len(features) > 0:
                mean_feature = np.mean(features[-1])
                # 特徴量の平均が負の値の場合、人狼の可能性が高いと仮定
                werewolf_prediction = bool(float(mean_feature) < -0.001)
                return werewolf_prediction
            return False
        
        try:
            if len(features) == 0:
                return False

            predictions = self.model.predict(features)

            # モデルの出力: 1=人狼, -1=その他
            # 複数行の中に1があれば人狼と判断
            werewolf_prediction = any(int(p) == 1 for p in predictions)

            return werewolf_prediction

        except Exception as e:
            print(f"Error in werewolf prediction: {e}")
            return False
    
    def check_and_judge(
        self, 
        info: Info, 
        current_agent: str, 
        status_tracker: Any
    ) -> None:
        """
        libsvmファイルをチェックして人狼判定を実行
        
        Args:
            info: ゲーム情報
            current_agent: 現在のエージェント名
            status_tracker: ステータストラッカーインスタンス
        """
        if not info or not info.game_id:
            return
        
        game_id = info.game_id
        game_timestamp = self.get_game_timestamp(game_id)
        
        # 処理済み数の初期化
        if game_id not in self.processed_counts:
            self.processed_counts[game_id] = {}
        if current_agent not in self.processed_counts[game_id]:
            self.processed_counts[game_id][current_agent] = {}
        
        # libsvmディレクトリをチェック
        libsvm_dir = Path("info") / "libsvm" / game_timestamp / current_agent
        if not libsvm_dir.exists():
            return
        
        # 各from_nameのlibsvmファイルをチェック
        for libsvm_file in libsvm_dir.glob("*.libsvm"):
            from_name = libsvm_file.stem
            
            # 現在の行数を取得
            current_count = self.count_libsvm_lines(libsvm_file)
            
            # 処理済み数を取得
            processed_count = self.processed_counts[game_id][current_agent].get(from_name, 0)
            
            # 4行以上かつ新しいデータがある場合に判定実行
            if current_count >= 4 and current_count > processed_count:
                print(f"Processing werewolf judgment for {from_name}: {current_count} lines")
                
                # libsvmデータを読み込み
                features = self.read_libsvm_data(libsvm_file)
                if features is not None:
                    # 人狼判定を実行
                    is_werewolf = self.predict_werewolf(features)
                    
                    # status.ymlに結果を反映
                    status_tracker.update_werewolf_status(from_name, is_werewolf)
                    
                    print(f"Werewolf judgment for {from_name}: {'werewolf' if is_werewolf else 'not werewolf'}")
                    
                    # 処理済み数を更新
                    self.processed_counts[game_id][current_agent][from_name] = current_count
    
    def reset_for_new_game(self, game_id: str) -> None:
        """
        新しいゲーム開始時に処理済み数をリセット
        
        Args:
            game_id: ゲームID
        """
        if game_id in self.processed_counts:
            del self.processed_counts[game_id]


def create_werewolf_judgment(config: Dict[str, Any]) -> WerewolfJudgment:
    """
    WerewolfJudgmentインスタンスを作成する便利関数
    
    Args:
        config: 設定辞書
        
    Returns:
        WerewolfJudgmentインスタンス
    """
    return WerewolfJudgment(config)


if __name__ == "__main__":
    # テスト用のサンプルコード
    pass