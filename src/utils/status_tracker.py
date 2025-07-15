"""エージェントの状態を追跡しstatus.ymlを生成するモジュール."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ulid import ULID, parse as ulid_parse

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk


class StatusTracker:
    """エージェントの状態を追跡するクラス."""

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
        self.packet_idx = 0
        
        # 状態履歴を保存する辞書
        # {packet_idx: {agent_name: {"self_co": ..., "seer_co": ..., "alive": ...}}}
        self.status_history: dict[int, dict[str, dict[str, Any]]] = {}
        
        # 現在の状態を保存する辞書
        self.current_status: dict[str, dict[str, Any]] = {}
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        ulid_obj: ULID = ulid_parse(self.game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(ulid_obj.timestamp().int / 1000, tz=tz).strftime(
            "%Y%m%d%H%M%S%f",
        )[:-3]
        
        # /info/status/game_timestamp/agent_name/status.yml の形式でディレクトリを作成
        self.output_dir = Path("info") / "status" / game_timestamp / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_file = self.output_dir / "status.yml"
    
    def update_status(
        self,
        info: Info,
        talk_history: list[Talk],
        self_co_results: dict[str, str | None],
        seer_co_results: dict[str, str | None],
    ) -> None:
        """エージェントの状態を更新."""
        self.packet_idx += 1
        
        # 各エージェントの状態を更新
        for agent_name, status in info.status_map.items():
            if agent_name not in self.current_status:
                self.current_status[agent_name] = {
                    "self_co": None,
                    "seer_co": None,
                    "alive": True,
                    "werewolf": None,
                }
            
            # self_coの更新（新しい値があれば更新）
            if agent_name in self_co_results and self_co_results[agent_name] is not None:
                self.current_status[agent_name]["self_co"] = self_co_results[agent_name]
            
            # seer_coの更新（新しい値があれば追加）
            if agent_name in seer_co_results and seer_co_results[agent_name] is not None:
                current_seer_co = self.current_status[agent_name]["seer_co"]
                new_seer_co = seer_co_results[agent_name]
                
                if current_seer_co is None:
                    self.current_status[agent_name]["seer_co"] = new_seer_co
                elif new_seer_co not in current_seer_co:
                    # カンマ区切りで追加
                    self.current_status[agent_name]["seer_co"] = f"{current_seer_co},{new_seer_co}"
            
            # 生存状態の更新
            self.current_status[agent_name]["alive"] = str(status) == "ALIVE"
        
        # 現在の状態を履歴に保存
        self.status_history[self.packet_idx] = {
            agent_name: dict(agent_status)
            for agent_name, agent_status in self.current_status.items()
        }
    
    def update_werewolf_status(self, agent_name: str, is_werewolf: bool) -> None:
        """
        特定のエージェントの人狼ステータスを更新
        
        Args:
            agent_name: エージェント名
            is_werewolf: 人狼かどうか（True/False）
        """
        # Python native boolに変換して保存
        werewolf_bool = bool(is_werewolf)
        
        if agent_name in self.current_status:
            self.current_status[agent_name]["werewolf"] = werewolf_bool
            
            # 現在のパケットインデックスの履歴も更新
            if self.packet_idx in self.status_history and agent_name in self.status_history[self.packet_idx]:
                self.status_history[self.packet_idx][agent_name]["werewolf"] = werewolf_bool
    
    def save_status(self) -> None:
        """status.ymlファイルに保存."""
        # YAMLフォーマットに変換
        yaml_data = {}
        for packet_idx, agents_status in self.status_history.items():
            yaml_data[packet_idx] = {}
            for agent_name, status in agents_status.items():
                # エージェント名をキーにして状態を保存
                yaml_data[packet_idx][f"agent    {agent_name.split('Agent')[1] if 'Agent' in agent_name else agent_name}"] = {
                    "self_co": status["self_co"] or "null",
                    "seer_co": status["seer_co"] or "null",
                    "alive": bool(status["alive"]),
                    "werewolf": status["werewolf"] if status["werewolf"] is not None else "null",
                }
        
        # YAMLファイルに書き込み
        with open(self.status_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False, default_style=None)