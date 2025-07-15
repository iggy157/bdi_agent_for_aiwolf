"""analysis.ymlから最適なsentenceを選択するモジュール."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ulid import ULID

if TYPE_CHECKING:
    pass


class SelectSentenceTracker:
    """analysis.ymlから最適なsentenceを選択して保存するクラス."""

    def __init__(
        self,
        agent_name: str,
        game_id: str,
    ) -> None:
        """初期化."""
        self.agent_name = agent_name
        self.game_id = game_id
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        ulid_obj: ULID = ULID.parse(self.game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(ulid_obj.timestamp / 1000, tz=tz).strftime(
            "%Y%m%d%H%M%S%f",
        )[:-3]
        
        # /info/select_sentence/game_timestamp/agent_name/select_sentence.yml の形式でディレクトリを作成
        self.output_dir = Path("info") / "select_sentence" / game_timestamp / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.select_sentence_file = self.output_dir / "select_sentence.yml"
        
        # analysis.ymlの場所を設定
        self.analysis_dir = Path("info") / "analysis" / game_timestamp / self.agent_name
        self.analysis_file = self.analysis_dir / "analysis.yml"
    
    def extract_all_entries_data(self) -> list[dict[str, Any]]:
        """analysis.ymlから全てのデータを抽出."""
        if not self.analysis_file.exists():
            return []
        
        try:
            with open(self.analysis_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                
                # カスタムフォーマットを解析（# Request N形式）
                entries = []
                current_entry = {}
                
                for line in content.split('\n'):
                    original_line = line
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    if line.endswith(':') and not original_line.startswith(' '):
                        # 新しいエントリの開始
                        if current_entry:
                            entries.append(current_entry)
                        current_entry = {}
                    elif original_line.startswith('  '):
                        # フィールドの解析
                        key_value = line.split(': ', 1)
                        if len(key_value) == 2:
                            key, value = key_value
                            if key == 'request_count':
                                current_entry[key] = int(value)
                            else:
                                current_entry[key] = value
                
                # 最後のエントリを追加
                if current_entry:
                    entries.append(current_entry)
                
                return entries
                
        except Exception:
            return []
    
    def select_sentence_by_request_count(self, entries: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        """各request_countごとにselect_sentenceの更新ルールに従ってデータを選択."""
        if not entries:
            return {}
        
        # request_countごとにエントリをグループ化
        entries_by_request_count = {}
        for entry in entries:
            request_count = entry.get('request_count', 0)
            if request_count not in entries_by_request_count:
                entries_by_request_count[request_count] = []
            entries_by_request_count[request_count].append(entry)
        
        # 各request_countごとに優先順位に従って選択
        selected_sentences = {}
        for request_count, request_entries in entries_by_request_count.items():
            selected_sentence = None
            found_self_target = False
            
            # エントリを順番にチェック
            for entry in request_entries:
                from_value = entry.get('from', '')
                to_value = entry.get('to', '')
                
                # from=自分のagent_name の場合はスキップ
                if from_value == self.agent_name:
                    continue
                
                # to=自分のagent_name の場合
                if to_value == self.agent_name:
                    selected_sentence = {
                        'content': entry.get('content', ''),
                        'request_count': request_count
                    }
                    found_self_target = True
                
                # to=all の場合（自分宛てが見つかっていない場合のみ）
                elif to_value == 'all' and not found_self_target:
                    selected_sentence = {
                        'content': entry.get('content', ''),
                        'request_count': request_count
                    }
            
            if selected_sentence:
                selected_sentences[request_count] = selected_sentence
        
        return selected_sentences
    
    def save_select_sentence(self, select_sentences: dict[int, dict[str, Any]]) -> None:
        """select_sentenceをYAML形式で保存."""
        try:
            # request_countの順番でソートして保存
            sorted_sentences = dict(sorted(select_sentences.items()))
            with open(self.select_sentence_file, "w", encoding="utf-8") as f:
                yaml.dump(sorted_sentences, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Failed to save select_sentence: {e}")
    
    def process_select_sentence(self) -> None:
        """メイン処理: analysis.ymlを解析してselect_sentence.ymlを生成."""
        # 1. 全てのデータを抽出
        all_entries = self.extract_all_entries_data()
        
        # 2. 各request_countごとに優先順位に従ってsentenceを選択
        selected_sentences = self.select_sentence_by_request_count(all_entries)
        
        # 3. 選択されたsentenceを保存
        if selected_sentences:
            self.save_select_sentence(selected_sentences)