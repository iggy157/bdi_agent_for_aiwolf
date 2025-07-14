"""エージェントの基底クラスを定義するモジュール."""

from __future__ import annotations

import os
import random
from pathlib import Path
from time import sleep
from typing import TYPE_CHECKING, Any, ParamSpec, TypeVar

import yaml
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
    from langchain_core.messages import BaseMessage

from aiwolf_nlp_common.packet import Info, Packet, Request, Role, Setting, Status, Talk

from utils.agent_logger import AgentLogger
from utils.stoppable_thread import StoppableThread
from utils.status_tracker import StatusTracker
from utils.co_extractor import COExtractor
from utils.analysis_tracker import AnalysisTracker
from utils.my_log_tracker import MyLogTracker
from utils.policy_evaluator import PolicyEvaluator
from utils.desire import DesireTracker

if TYPE_CHECKING:
    from collections.abc import Callable

P = ParamSpec("P")
T = TypeVar("T")


class Agent:
    """エージェントの基底クラス."""

    def __init__(
        self,
        config: dict[str, Any],
        name: str,
        game_id: str,
        role: Role,
    ) -> None:
        """エージェントの初期化を行う."""
        self.config = config
        self.agent_name = name
        self.agent_logger = AgentLogger(config, name, game_id)
        self.request: Request | None = None
        self.info: Info | None = None
        self.setting: Setting | None = None
        self.talk_history: list[Talk] = []
        self.whisper_history: list[Talk] = []
        self.role = role

        self.sent_talk_count: int = 0
        self.sent_whisper_count: int = 0
        self.llm_model: BaseChatModel | None = None
        self.llm_message_history: list[BaseMessage] = []
        
        # Request counter for analysis tracking
        self.total_request_count: int = 0
        
        # Status tracking
        self.status_tracker = StatusTracker(config, name, game_id)
        self.co_extractor = COExtractor(config)
        
        # Analysis tracking
        self.analysis_tracker = AnalysisTracker(config, name, game_id)
        
        # My log tracking
        self.my_log_tracker = MyLogTracker(config, name, game_id)
        
        # Policy and desire tracking
        self.policy_evaluator = PolicyEvaluator()
        self.desire_tracker = DesireTracker(config)
        self.game_id = game_id

        load_dotenv(Path(__file__).parent.joinpath("./../../config/.env"))

    @staticmethod
    def timeout(func: Callable[P, T]) -> Callable[P, T]:
        """アクションタイムアウトを設定するデコレータ."""

        def _wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            res: T | Exception = Exception("No result")

            def execute_with_timeout() -> None:
                nonlocal res
                try:
                    res = func(*args, **kwargs)
                except Exception as e:  # noqa: BLE001
                    res = e

            thread = StoppableThread(target=execute_with_timeout)
            thread.start()
            self = args[0] if args else None
            if not isinstance(self, Agent):
                raise TypeError(self, " is not an Agent instance")
            timeout_value = (self.setting.timeout.action if hasattr(self, "setting") and self.setting else 0) // 1000
            if timeout_value > 0:
                thread.join(timeout=timeout_value)
                if thread.is_alive():
                    self.agent_logger.logger.warning(
                        "アクションがタイムアウトしました: %s",
                        self.request,
                    )
                    if bool(self.config["agent"]["kill_on_timeout"]):
                        thread.stop()
                        self.agent_logger.logger.warning(
                            "アクションを強制終了しました: %s",
                            self.request,
                        )
            else:
                thread.join()
            if isinstance(res, Exception):  # type: ignore[arg-type]
                raise res
            return res

        return _wrapper

    def set_packet(self, packet: Packet) -> None:
        """パケット情報をセットする."""
        self.request = packet.request
        if packet.info:
            self.info = packet.info
        if packet.setting:
            self.setting = packet.setting
        if packet.talk_history:
            self.talk_history.extend(packet.talk_history)
        if packet.whisper_history:
            self.whisper_history.extend(packet.whisper_history)
        if self.request == Request.INITIALIZE:
            self.talk_history: list[Talk] = []
            self.whisper_history: list[Talk] = []
            self.llm_message_history: list[BaseMessage] = []
            self.total_request_count = 0  # リセット
        else:
            # リクエストカウンターをインクリメント
            self.total_request_count += 1
        self.agent_logger.logger.debug(packet)
        
        # Update status tracking if we have info
        if self.info and self.talk_history:
            self._update_status_tracking()

    def get_alive_agents(self) -> list[str]:
        """生存しているエージェントのリストを取得する."""
        if not self.info:
            return []
        return [k for k, v in self.info.status_map.items() if v == Status.ALIVE]

    def _send_message_to_llm(self, request: Request | None) -> str | None:
        if request is None:
            return None
        if request.lower() not in self.config["prompt"]:
            return None
        prompt = self.config["prompt"][request.lower()]
        if float(self.config["llm"]["sleep_time"]) > 0:
            sleep(float(self.config["llm"]["sleep_time"]))
        key = {
            "info": self.info,
            "setting": self.setting,
            "talk_history": self.talk_history,
            "whisper_history": self.whisper_history,
            "role": self.role,
            "sent_talk_count": self.sent_talk_count,
            "sent_whisper_count": self.sent_whisper_count,
            "intention_data": self._get_latest_intention_data(),
        }
        template: Template = Template(prompt)
        prompt = template.render(**key).strip()
        if self.llm_model is None:
            self.agent_logger.logger.error("LLM is not initialized")
            return None
        try:
            self.llm_message_history.append(HumanMessage(content=prompt))
            response = (self.llm_model | StrOutputParser()).invoke(self.llm_message_history)
            self.llm_message_history.append(AIMessage(content=response))
            self.agent_logger.logger.info(["LLM", prompt, response])
        except Exception:
            self.agent_logger.logger.exception("Failed to send message to LLM")
            return None
        else:
            return response

    @timeout
    def name(self) -> str:
        """名前リクエストに対する応答を返す."""
        return self.agent_name

    def initialize(self) -> None:
        """ゲーム開始リクエストに対する初期化処理を行う."""
        if self.info is None:
            return

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
        self.llm_model = self.llm_model
        self._send_message_to_llm(self.request)

    def daily_initialize(self) -> None:
        """昼開始リクエストに対する処理を行う."""
        self._send_message_to_llm(self.request)

    def whisper(self) -> str:
        """囁きリクエストに対する応答を返す."""
        response = self._send_message_to_llm(self.request)
        self.sent_whisper_count = len(self.whisper_history)
        
        # 発話内容を記録
        if response:
            self.my_log_tracker.log_talk(response, "whisper", self.info)
        
        return response or ""

    def talk(self) -> str:
        """トークリクエストに対する応答を返す."""
        # ポリシー評価とdesire生成を実行
        self._evaluate_policy_and_generate_desire()
        
        response = self._send_message_to_llm(self.request)
        self.sent_talk_count = len(self.talk_history)
        
        # 発話内容を記録
        if response:
            self.my_log_tracker.log_talk(response, "talk", self.info)
        
        return response or ""

    def daily_finish(self) -> None:
        """昼終了リクエストに対する処理を行う."""
        self._send_message_to_llm(self.request)

    def divine(self) -> str:
        """占いリクエストに対する応答を返す."""
        response = self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )
        
        # アクション内容を記録
        if response:
            self.my_log_tracker.log_action("divine", response, self.info)
        
        return response

    def guard(self) -> str:
        """護衛リクエストに対する応答を返す."""
        response = self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )
        
        # アクション内容を記録
        if response:
            self.my_log_tracker.log_action("guard", response, self.info)
        
        return response

    def vote(self) -> str:
        """投票リクエストに対する応答を返す."""
        response = self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )
        
        # アクション内容を記録
        if response:
            self.my_log_tracker.log_action("vote", response, self.info)
        
        return response

    def attack(self) -> str:
        """襲撃リクエストに対する応答を返す."""
        response = self._send_message_to_llm(self.request) or random.choice(  # noqa: S311
            self.get_alive_agents(),
        )
        
        # アクション内容を記録
        if response:
            self.my_log_tracker.log_action("attack", response, self.info)
        
        return response

    def finish(self) -> None:
        """ゲーム終了リクエストに対する処理を行う."""
        # Save status.yml when game finishes
        self.status_tracker.save_status()
        
        # Save analysis.yml when game finishes
        self.analysis_tracker.save_analysis()
        
        # Save my_log.yml when game finishes
        self.my_log_tracker.save_my_log()
    
    def _update_status_tracking(self) -> None:
        """ステータストラッキングを更新."""
        if not self.info:
            return
        
        try:
            # Extract self_co from talk history
            self_co_results = self.co_extractor.extract_self_co(self.talk_history)
            
            # Extract seer_co from talk history
            seer_co_results = self.co_extractor.extract_seer_co(self.talk_history)
            
            # Update status tracker
            self.status_tracker.update_status(
                self.info,
                self.talk_history,
                self_co_results,
                seer_co_results,
            )
            
            # Save status after each update
            self.status_tracker.save_status()
            
            # Update analysis tracker with current request count
            self.analysis_tracker.analyze_talk(self.talk_history, self.info, self.total_request_count)
            
            # Save analysis after each update
            self.analysis_tracker.save_analysis()
            
            # Save my_log after each update
            self.my_log_tracker.save_my_log()
        except Exception as e:
            self.agent_logger.logger.error(f"Failed to update status tracking: {e}")

    def _evaluate_policy_and_generate_desire(self) -> None:
        """ポリシー評価とdesire生成を実行."""
        if not self.info:
            return
        
        try:
            # エージェントの役職名を取得
            role_name = self.role.value.lower()
            
            # ポリシールールを評価して条件に一致しないルールを取得
            best_policy = self.policy_evaluator.evaluate_policy_rules(
                role_name, 
                self.game_id, 
                self.agent_name
            )
            
            # best_policyからdesire文を生成して保存
            if best_policy:
                self.desire_tracker.generate_and_save_desire(
                    best_policy,
                    self.game_id,
                    self.agent_name,
                    self.info
                )
                self.agent_logger.logger.info(f"Generated desire for {self.agent_name}")
            
        except Exception as e:
            self.agent_logger.logger.error(f"Failed to evaluate policy and generate desire: {e}")

    @timeout
    def action(self) -> str | None:  # noqa: C901, PLR0911
        """リクエストの種類に応じたアクションを実行する."""
        match self.request:
            case Request.NAME:
                return self.name()
            case Request.TALK:
                return self.talk()
            case Request.WHISPER:
                return self.whisper()
            case Request.VOTE:
                return self.vote()
            case Request.DIVINE:
                return self.divine()
            case Request.GUARD:
                return self.guard()
            case Request.ATTACK:
                return self.attack()
            case Request.INITIALIZE:
                self.initialize()
            case Request.DAILY_INITIALIZE:
                self.daily_initialize()
            case Request.DAILY_FINISH:
                self.daily_finish()
            case Request.FINISH:
                self.finish()
            case _:
                pass
        return None

    def _get_latest_intention_data(self) -> dict[str, Any] | None:
        """最新のintention data（request_countが最大）を取得."""
        if not self.info or not self.info.game_id:
            return None
        
        try:
            from ulid import ULID
            from datetime import UTC, datetime
            
            # ULIDからタイムスタンプを取得
            ulid_obj = ULID.parse(self.info.game_id)
            tz = datetime.now(UTC).astimezone().tzinfo
            game_timestamp = datetime.fromtimestamp(ulid_obj.timestamp / 1000, tz=tz).strftime(
                "%Y%m%d%H%M%S%f",
            )[:-3]
            
            # intentionファイルのパス
            intention_file = Path("info") / "intention" / game_timestamp / self.agent_name / "intention.yml"
            
            if not intention_file.exists():
                return None
            
            # intentionファイルを読み込み
            with open(intention_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if not data:
                return None
            
            # 最新のrequest_count（最大値）のデータを取得
            max_request_count = max(int(key) if isinstance(key, str) and key.isdigit() else key for key in data.keys())
            latest_intention = data.get(max_request_count)
            
            if latest_intention:
                return {
                    "request_count": max_request_count,
                    "consist": latest_intention.get("consist", ""),
                    "content": latest_intention.get("content", "")
                }
            
            return None
            
        except Exception as e:
            self.agent_logger.logger.warning(f"Failed to load intention data: {e}")
            return None
