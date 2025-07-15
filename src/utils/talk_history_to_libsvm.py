#!/usr/bin/env python3
"""
Talk履歴からlibsvm形式への変換モジュール

サーバから送られてくるtalk_historyを処理して、各エージェントの発言を
Word2Vec+MeCabでlibsvm形式に変換し、指定されたディレクトリ構造に保存・蓄積する。

ディレクトリ構造：
/info/libsvm/game_name/agent_name/from_name.libsvm

Example:
    /info/libsvm/202507101349/agent1/ミナト.libsvm
    /info/libsvm/202507101349/agent1/タクミ.libsvm
    /info/libsvm/202507101349/agent1/ミサキ.libsvm
    /info/libsvm/202507101349/agent1/リン.libsvm
"""

from __future__ import annotations

import warnings
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Optional
from collections import defaultdict
import numpy as np
from datetime import datetime

warnings.filterwarnings('ignore')

# 必須: pip install gensim mecab-python3
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

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Talk, Info


class TalkHistoryToLibsvmConverter:
    """Talk履歴をlibsvm形式に変換するクラス"""
    
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, epochs: int = 10):
        """
        初期化
        
        Args:
            vector_size: Word2Vecの次元数
            window: Word2Vecのウィンドウサイズ
            min_count: Word2Vecの最小出現回数
            epochs: Word2Vecの学習エポック数
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.mecab = MeCab.Tagger("-Owakati") if MECAB_AVAILABLE else None
    
    def tokenize_japanese(self, text: str) -> List[str]:
        """
        日本語テキストをMeCabで分かち書きする
        
        Args:
            text: 分かち書きするテキスト
            
        Returns:
            分かち書きされたトークンのリスト
        """
        if self.mecab:
            try:
                return [t for t in self.mecab.parse(text).strip().split() if t]
            except Exception:
                pass
        # Fallback: 文字列から日本語単語っぽい部分を切り出す
        return re.findall(r'[ぁ-んァ-ン一-龥a-zA-Z0-9]+', text)
    
    def extract_talks_by_agent(self, talk_history: List[Talk], current_agent: str) -> Dict[str, List[str]]:
        """
        talk_historyから現在のエージェント以外のエージェントの発言を抽出
        
        Args:
            talk_history: トーク履歴のリスト
            current_agent: 現在のエージェント名
            
        Returns:
            エージェント名をキーとし、発言のリストを値とする辞書
        """
        talks_by_agent = defaultdict(list)
        
        for talk in talk_history:
            # 現在のエージェント以外の発言のみを対象とする
            if talk.agent != current_agent:
                talks_by_agent[talk.agent].append(talk.text)
        
        return dict(talks_by_agent)
    
    def train_word2vec(self, all_texts: List[str]) -> Optional[Word2Vec]:
        """
        テキストリストからWord2Vecモデルを学習
        
        Args:
            all_texts: 学習用テキストのリスト
            
        Returns:
            学習済みWord2Vecモデル
        """
        if not GENSIM_AVAILABLE:
            print("Error: gensim not available.")
            return None
        
        sentences = [self.tokenize_japanese(text) for text in all_texts if text.strip()]
        if not sentences:
            print("Warning: No content found for Word2Vec training")
            return None
        
        try:
            model = Word2Vec(
                sentences, 
                vector_size=self.vector_size, 
                window=self.window,
                min_count=self.min_count, 
                epochs=self.epochs
            )
            return model
        except Exception as e:
            print(f"Error training Word2Vec model: {e}")
            return None
    
    def text_to_vector(self, text: str, model: Word2Vec) -> np.ndarray:
        """
        テキストをWord2Vec平均ベクトルに変換
        
        Args:
            text: 変換するテキスト
            model: Word2Vecモデル
            
        Returns:
            平均ベクトル
        """
        tokens = self.tokenize_japanese(text)
        vecs = [model.wv[token] for token in tokens if token in model.wv]
        if vecs:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(model.vector_size)
    
    def vector_to_libsvm(self, vector: np.ndarray, label: int = 1) -> str:
        """
        ベクトルをlibsvm形式の1行に変換
        
        Args:
            vector: 変換するベクトル
            label: ラベル（デフォルト: 1）
            
        Returns:
            libsvm形式の文字列
        """
        features = [f"{i+1}:{x:.6f}" for i, x in enumerate(vector) if abs(x) > 1e-6]
        return f"{label} " + " ".join(features)
    
    def save_libsvm_to_file(self, libsvm_lines: List[str], file_path: Path) -> None:
        """
        libsvm形式のデータをファイルに保存（追記モード）
        
        Args:
            libsvm_lines: libsvm形式の行のリスト
            file_path: 保存先ファイルパス
        """
        # ディレクトリを作成
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 追記モードでファイルに保存
        with open(file_path, "a", encoding="utf-8") as f:
            for line in libsvm_lines:
                f.write(line + "\n")
    
    def get_game_timestamp(self, info: Info) -> str:
        """
        infoオブジェクトからゲーム開始時刻のタイムスタンプを取得
        
        Args:
            info: ゲーム情報
            
        Returns:
            タイムスタンプ文字列
        """
        if info and info.game_id:
            try:
                from ulid import ULID, parse as ulid_parse
                from datetime import UTC, datetime
                
                # ULIDからタイムスタンプを取得
                ulid_obj = ulid_parse(info.game_id)
                tz = datetime.now(UTC).astimezone().tzinfo
                game_timestamp = datetime.fromtimestamp(ulid_obj.timestamp().int / 1000, tz=tz).strftime(
                    "%Y%m%d%H%M%S%f",
                )[:-3]
                return game_timestamp
            except Exception:
                pass
        
        # フォールバック: 現在時刻を使用
        return datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]
    
    def process_talk_history(
        self, 
        talk_history: List[Talk], 
        current_agent: str, 
        info: Info
    ) -> None:
        """
        talk_historyを処理してlibsvmファイルを生成・蓄積
        
        Args:
            talk_history: トーク履歴のリスト
            current_agent: 現在のエージェント名
            info: ゲーム情報
        """
        if not talk_history:
            return
        
        # ゲームタイムスタンプを取得
        game_timestamp = self.get_game_timestamp(info)
        
        # 現在のエージェント以外の発言を抽出
        talks_by_agent = self.extract_talks_by_agent(talk_history, current_agent)
        
        if not talks_by_agent:
            return
        
        # 全ての発言テキストを収集してWord2Vecモデルを学習
        all_texts = []
        for agent_talks in talks_by_agent.values():
            all_texts.extend(agent_talks)
        
        model = self.train_word2vec(all_texts)
        if model is None:
            return
        
        # エージェントごとにlibsvmファイルを生成・蓄積
        for from_agent, texts in talks_by_agent.items():
            libsvm_lines = []
            for text in texts:
                vector = self.text_to_vector(text, model)
                libsvm_line = self.vector_to_libsvm(vector)
                libsvm_lines.append(libsvm_line)
            
            # ファイルパスを構築
            file_path = Path("info") / "libsvm" / game_timestamp / current_agent / f"{from_agent}.libsvm"
            
            # ファイルに保存（追記）
            self.save_libsvm_to_file(libsvm_lines, file_path)
            
            print(f"Saved libsvm data: {file_path} ({len(libsvm_lines)} lines)")


def process_server_request_to_libsvm(
    talk_history: List[Talk], 
    current_agent: str, 
    info: Info,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 1,
    epochs: int = 10
) -> None:
    """
    サーバリクエストのtalk_historyを処理してlibsvmファイルを生成する便利関数
    
    Args:
        talk_history: トーク履歴のリスト
        current_agent: 現在のエージェント名
        info: ゲーム情報
        vector_size: Word2Vecの次元数
        window: Word2Vecのウィンドウサイズ
        min_count: Word2Vecの最小出現回数
        epochs: Word2Vecの学習エポック数
    """
    converter = TalkHistoryToLibsvmConverter(vector_size, window, min_count, epochs)
    converter.process_talk_history(talk_history, current_agent, info)


if __name__ == "__main__":
    # テスト用のサンプルコード
    pass