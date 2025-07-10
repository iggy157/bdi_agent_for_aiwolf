"""自エージェントの発話履歴を記録するmy_log.ymlを生成するモジュール."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ulid import ULID

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Request


class MyLogTracker:
    """自エージェントの発話履歴を記録するクラス."""

    def __init__(
        self,
        config: dict[str, Any],
        agent_name: str,
        game_id: str,
    ) -> None:
        """初期化."""
        self.config = config
        self.agent_name = agent_name
        self.game_id = game_id
        
        # 発話履歴を保存するリスト
        self.my_log_history: list[dict[str, Any]] = []
        
        # 発話カウンター
        self.talk_counter = 0
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        ulid: ULID = ULID.from_str(self.game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime(
            "%Y%m%d%H%M%S%f",
        )[:-3]
        
        # /info/my_log/game_timestamp/agent_name/my_log.yml の形式でディレクトリを作成
        self.output_dir = Path("info") / "my_log" / game_timestamp / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.my_log_file = self.output_dir / "my_log.yml"
    
    def log_talk(
        self,
        talk_content: str,
        request_type: str,
        info: Info | None = None,
    ) -> None:
        """発話内容を記録."""
        if not talk_content:
            return
        
        self.talk_counter += 1
        
        # 現在の時刻を記録
        current_time = datetime.now(UTC).isoformat()
        
        # 発話エントリを作成
        log_entry = {
            "id": self.talk_counter,
            "timestamp": current_time,
            "type": request_type,  # "talk" または "whisper"
            "content": talk_content,
            "agent": self.agent_name,
        }
        
        # ゲーム情報が利用可能な場合は追加
        if info:
            log_entry["day"] = info.day
            log_entry["game_id"] = info.game_id
        
        # 履歴に追加
        self.my_log_history.append(log_entry)
    
    def log_action(
        self,
        action_type: str,
        action_content: str,
        info: Info | None = None,
    ) -> None:
        """アクション（投票、占い、護衛、襲撃など）を記録."""
        if not action_content:
            return
        
        self.talk_counter += 1
        
        # 現在の時刻を記録
        current_time = datetime.now(UTC).isoformat()
        
        # アクションエントリを作成
        log_entry = {
            "id": self.talk_counter,
            "timestamp": current_time,
            "type": action_type,  # "vote", "divine", "guard", "attack"
            "content": action_content,
            "agent": self.agent_name,
        }
        
        # ゲーム情報が利用可能な場合は追加
        if info:
            log_entry["day"] = info.day
            log_entry["game_id"] = info.game_id
        
        # 履歴に追加
        self.my_log_history.append(log_entry)
    
    def save_my_log(self) -> None:
        """my_log.ymlファイルに保存."""
        # YAMLフォーマットに変換
        yaml_data = {}
        
        for entry in self.my_log_history:
            yaml_data[entry["id"]] = {
                "timestamp": entry["timestamp"],
                "type": entry["type"],
                "content": entry["content"],
                "agent": entry["agent"],
            }
            
            # ゲーム情報がある場合は追加
            if "day" in entry:
                yaml_data[entry["id"]]["day"] = entry["day"]
            if "game_id" in entry:
                yaml_data[entry["id"]]["game_id"] = entry["game_id"]
        
        # YAMLファイルに書き込み
        with open(self.my_log_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def get_talk_count(self) -> int:
        """発話総数を取得."""
        return len([entry for entry in self.my_log_history if entry["type"] in ["talk", "whisper"]])
    
    def get_action_count(self) -> int:
        """アクション総数を取得."""
        return len([entry for entry in self.my_log_history if entry["type"] in ["vote", "divine", "guard", "attack"]])
    
    def get_total_count(self) -> int:
        """総エントリ数を取得."""
        return len(self.my_log_history)