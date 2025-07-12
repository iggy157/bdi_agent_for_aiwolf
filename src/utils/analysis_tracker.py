"""analysis.ymlを生成するためのモジュール."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ulid import ULID

if TYPE_CHECKING:
    from aiwolf_nlp_common.packet import Info, Talk

from .co_extractor import COExtractor


class AnalysisTracker:
    """トーク分析を行いanalysis.ymlを生成するクラス."""

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
        
        # 分析履歴を保存する辞書
        # {packet_idx: [{"content": ..., "type": ..., "from": ..., "to": ..., "request_count": ...}, ...]}
        self.analysis_history: dict[int, list[dict[str, Any]]] = {}
        
        # 前回分析したトークの数を記録
        self.last_analyzed_talk_count = 0
        
        # リクエストカウンターを初期化
        self.request_count = 0
        
        # LLMクライアントの初期化
        self.llm_client = COExtractor(config)
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        ulid: ULID = ULID.from_str(self.game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(ulid.timestamp, tz=tz).strftime(
            "%Y%m%d%H%M%S%f",
        )[:-3]
        
        # /info/analysis/game_timestamp/agent_name/analysis.yml の形式でディレクトリを作成
        self.output_dir = Path("info") / "analysis" / game_timestamp / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_file = self.output_dir / "analysis.yml"
    
    def analyze_talk(
        self,
        talk_history: list[Talk],
        info: Info,
        request_count: int | None = None,
    ) -> None:
        """トーク履歴を分析してanalysis.ymlを更新."""
        # 新しいトーク履歴のエントリだけを分析
        new_talks = talk_history[self.last_analyzed_talk_count:]
        
        if not new_talks:
            return  # 新しいトークがない場合は何もしない
        
        # request_countが渡された場合は更新
        if request_count is not None:
            self.request_count = request_count
        
        self.packet_idx += 1
        
        # 新しいトーク履歴のエントリを分析
        analysis_entries = []
        
        for talk in new_talks:
            # 発話内容の分析
            analysis_entry = self._analyze_talk_entry(talk, info)
            if analysis_entry:
                analysis_entries.append(analysis_entry)
        
        # 分析結果を履歴に保存
        if analysis_entries:
            self.analysis_history[self.packet_idx] = analysis_entries
        
        # 分析済みトーク数を更新
        self.last_analyzed_talk_count = len(talk_history)
    
    def _analyze_talk_entry(
        self,
        talk: Talk,
        info: Info,
    ) -> dict[str, Any] | None:
        """個別のトーク発話を分析."""
        if not talk.text or talk.text.strip() == "":
            return None
        
        # "Over"発話は分析対象に含める
        if talk.text.strip() == "Over":
            return {
                "content": talk.text,
                "type": "null",
                "from": talk.agent,
                "to": "null",
                "request_count": self.request_count,
            }
        
        # 基本情報を抽出
        content = talk.text
        from_agent = talk.agent
        
        # LLMを使って発話を分析
        message_type = self._analyze_message_type(content, info)
        to_agents = self._analyze_target_agents(content, info)
        
        return {
            "content": content,
            "type": message_type,
            "from": from_agent,
            "to": to_agents,
            "request_count": self.request_count,
        }
    
    def _analyze_message_type(
        self,
        content: str,
        info: Info,
    ) -> str:
        """発話のタイプを分析."""
        # LLMを使って発話タイプを分析
        if not self.llm_client.llm_model:
            return "null"
        
        try:
            from jinja2 import Template
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # プロンプトテンプレートの作成
            prompt_template = self.config.get("prompt", {}).get("analyze_message_type", "")
            if not prompt_template:
                return "null"
            
            template = Template(prompt_template)
            prompt = template.render(
                content=content,
                agent_names=list(info.status_map.keys()),
            )
            
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_client.llm_model | StrOutputParser()).invoke(messages)
            
            # 応答からタイプを抽出
            response = response.strip().lower()
            
            # 優先順位順でタイプを判定
            if "co" in response:
                return "co"
            elif "question" in response:
                return "question"
            elif "negative" in response:
                return "negative"
            elif "positive" in response:
                return "positive"
            else:
                return "null"
                
        except Exception:
            return "null"
    
    def _analyze_target_agents(
        self,
        content: str,
        info: Info,
    ) -> str:
        """発話の対象エージェントを分析."""
        # LLMを使って発話対象を分析
        if not self.llm_client.llm_model:
            return "null"
        
        try:
            from jinja2 import Template
            from langchain_core.messages import HumanMessage
            from langchain_core.output_parsers import StrOutputParser
            
            # プロンプトテンプレートの作成
            prompt_template = self.config.get("prompt", {}).get("analyze_target_agents", "")
            if not prompt_template:
                return "null"
            
            template = Template(prompt_template)
            prompt = template.render(
                content=content,
                agent_names=list(info.status_map.keys()),
            )
            
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_client.llm_model | StrOutputParser()).invoke(messages)
            
            # 応答から対象エージェントを抽出
            response = response.strip()
            
            # 特定のエージェント名が含まれているかチェック
            mentioned_agents = []
            for agent_name in info.status_map.keys():
                if agent_name in response:
                    mentioned_agents.append(agent_name)
            
            if mentioned_agents:
                return ",".join(mentioned_agents)
            elif "all" in response.lower() or "全体" in response:
                return "all"
            else:
                return "null"
                
        except Exception:
            return "null"
    
    def save_analysis(self) -> None:
        """analysis.ymlファイルに保存."""
        # request_countごとにグループ化
        request_groups: dict[int, list[dict[str, Any]]] = {}
        
        # 全エントリをrequest_countでグループ化
        for packet_idx, entries in self.analysis_history.items():
            for entry in entries:
                request_count = entry.get("request_count", 0)
                if request_count not in request_groups:
                    request_groups[request_count] = []
                request_groups[request_count].append(entry)
        
        # YAMLファイルに書き込み（カスタムフォーマット）
        with open(self.analysis_file, "w", encoding="utf-8") as f:
            entry_counter = 1
            for request_count in sorted(request_groups.keys()):
                # リクエストヘッダーを書き込み
                f.write(f"# Request {request_count}\n")
                
                # そのリクエストの全エントリを書き込み
                for i, entry in enumerate(request_groups[request_count]):
                    f.write(f"{entry_counter}:\n")
                    f.write(f"  content: {entry['content']}\n")
                    f.write(f"  type: {entry['type']}\n")
                    f.write(f"  from: {entry['from']}\n")
                    f.write(f"  to: {entry['to']}\n")
                    f.write(f"  request_count: {entry.get('request_count', 0)}\n")
                    if i < len(request_groups[request_count]) - 1:
                        f.write("\n")
                    entry_counter += 1
                
                f.write("\n\n")