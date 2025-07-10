"""トーク履歴から役職のカミングアウト情報を抽出するモジュール."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

from aiwolf_nlp_common.packet import Talk


class COExtractor:
    """カミングアウト情報を抽出するクラス."""

    def __init__(self, config: dict[str, Any]) -> None:
        """初期化."""
        self.config = config
        self.llm_model: BaseChatModel | None = None
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """LLMモデルの初期化."""
        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))
        
        model_type = str(self.config["llm"]["type"])
        match model_type:
            case "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            case "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            case "ollama":
                self.llm_model = ChatOllama(
                    model=str(self.config["ollama"]["model"]),
                    temperature=float(self.config["ollama"]["temperature"]),
                    base_url=str(self.config["ollama"]["base_url"]),
                )
            case _:
                raise ValueError(model_type, "Unknown LLM type")
    
    def extract_self_co(self, talk_history: list[Talk]) -> dict[str, str | None]:
        """各エージェントのself_co（自己申告の役職）を抽出."""
        if not self.llm_model:
            return {}
        
        results: dict[str, str | None] = {}
        
        # 各エージェントごとにトーク履歴を分析
        agents = {talk.agent for talk in talk_history}
        for agent in agents:
            agent_talks = [talk for talk in talk_history if talk.agent == agent]
            
            if not agent_talks:
                results[agent] = None
                continue
            
            # プロンプトの作成
            prompt_template = self.config["prompt"]["extract_self_co"]
            template = Template(prompt_template)
            prompt = template.render(
                agent_name=agent,
                talk_history=[{"agent": talk.agent, "text": talk.text} for talk in agent_talks],
            )
            
            try:
                # LLMに問い合わせ
                messages = [HumanMessage(content=prompt)]
                response = (self.llm_model | StrOutputParser()).invoke(messages)
                
                # 役職を抽出
                role = self._parse_role_from_response(response)
                results[agent] = role
            except Exception:
                results[agent] = None
        
        return results
    
    def extract_seer_co(self, talk_history: list[Talk]) -> dict[str, str | None]:
        """占い師によるseer_co（占い結果）を抽出."""
        if not self.llm_model:
            return {}
        
        results: dict[str, str | None] = {}
        
        # プロンプトの作成
        prompt_template = self.config["prompt"]["extract_seer_co"]
        template = Template(prompt_template)
        prompt = template.render(
            talk_history=[{"agent": talk.agent, "text": talk.text} for talk in talk_history],
        )
        
        try:
            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            response = (self.llm_model | StrOutputParser()).invoke(messages)
            
            # 占い結果を解析
            results = self._parse_seer_results_from_response(response)
        except Exception:
            pass
        
        return results
    
    def _parse_role_from_response(self, response: str) -> str | None:
        """LLMの応答から役職を解析."""
        response = response.strip().lower()
        
        # 役職のマッピング
        role_mapping = {
            "村人": "村人",
            "villager": "村人",
            "占い師": "占い師",
            "seer": "占い師",
            "霊能者": "霊能者",
            "霊媒師": "霊能者",
            "medium": "霊能者",
            "人狼": "人狼",
            "werewolf": "人狼",
            "狂人": "狂人",
            "possessed": "狂人",
            "狩人": "狩人",
            "騎士": "狩人",
            "bodyguard": "狩人",
        }
        
        for key, value in role_mapping.items():
            if key in response:
                return value
        
        return None
    
    def _parse_seer_results_from_response(self, response: str) -> dict[str, str | None]:
        """LLMの応答から占い結果を解析."""
        results: dict[str, str | None] = {}
        
        # 簡単な解析ロジック（改善の余地あり）
        lines = response.strip().split("\n")
        for line in lines:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    agent_name = parts[0].strip()
                    result = parts[1].strip()
                    
                    # 人狼か人間かの判定
                    if "人狼" in result or "werewolf" in result.lower():
                        results[agent_name] = "人狼"
                    elif "人間" in result or "human" in result.lower():
                        results[agent_name] = "人間"
        
        return results