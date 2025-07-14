"""エージェントのintentionを生成し保存するモジュール."""

from __future__ import annotations

import os
import yaml
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from ulid import ULID

if TYPE_CHECKING:
    pass


class IntentionTracker:
    """エージェントのintentionを生成し保存するクラス."""

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
        self.llm_model = None
        
        # .envファイルを読み込み
        load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
        
        # LLMモデルを初期化
        self._initialize_llm()
        
        # 出力ディレクトリの設定
        self._setup_output_directory()
    
    def _setup_output_directory(self) -> None:
        """出力ディレクトリの設定."""
        ulid_obj: ULID = ULID.parse(self.game_id)
        tz = datetime.now(UTC).astimezone().tzinfo
        game_timestamp = datetime.fromtimestamp(ulid_obj.timestamp / 1000, tz=tz).strftime(
            "%Y%m%d%H%M%S%f",
        )[:-3]
        
        # /info/intention/game_timestamp/agent_name/intention.yml の形式でディレクトリを作成
        self.output_dir = Path("info") / "intention" / game_timestamp / self.agent_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.intention_file = self.output_dir / "intention.yml"
        
        # 入力データファイルのパス設定
        self.game_timestamp = game_timestamp
        self.analysis_file = Path("info") / "analysis" / game_timestamp / self.agent_name / "analysis.yml"
        self.select_sentence_file = Path("info") / "select_sentence" / game_timestamp / self.agent_name / "select_sentence.yml"
        self.desire_file = Path("info") / "desire" / self.game_id / self.agent_name / "desire.yml"
        self.status_file = Path("info") / "status" / game_timestamp / self.agent_name / "status.yml"
        self.my_log_file = Path("info") / "my_log" / game_timestamp / self.agent_name / "my_log.yml"

    def _initialize_llm(self) -> None:
        """LLMモデルを初期化する."""
        try:
            model_type = str(self.config["llm"]["type"])
            
            if model_type == "openai":
                self.llm_model = ChatOpenAI(
                    model=str(self.config["openai"]["model"]),
                    temperature=float(self.config["openai"]["temperature"]),
                    api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
                )
            elif model_type == "google":
                self.llm_model = ChatGoogleGenerativeAI(
                    model=str(self.config["google"]["model"]),
                    temperature=float(self.config["google"]["temperature"]),
                    google_api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            else:
                raise ValueError(f"Unsupported LLM type: {model_type}")
                
        except Exception as e:
            print(f"Failed to initialize LLM: {e}")
            self.llm_model = None

    def _load_yaml_file(self, file_path: Path) -> dict[str, Any] | None:
        """YAMLファイルを読み込む."""
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load {file_path}: {e}")
            return None

    def _get_latest_select_sentence(self) -> dict[str, Any] | None:
        """select_sentence.ymlから最新データ（request_countが最大）を取得."""
        data = self._load_yaml_file(self.select_sentence_file)
        if not data:
            return None
        
        # request_countが最大のデータを取得
        max_request_count = max(int(key) if isinstance(key, str) and key.isdigit() else key for key in data.keys())
        return data.get(max_request_count)

    def _get_latest_desire(self) -> dict[str, Any] | None:
        """desire.ymlから最新データ（request_countが最大）を取得."""
        data = self._load_yaml_file(self.desire_file)
        if not data or 'desires' not in data:
            return None
        
        desires = data['desires']
        if not desires:
            return None
        
        # request_countが最大のデータを取得
        latest_desire = max(desires, key=lambda x: x.get('request_count', 0))
        return latest_desire

    def _get_analysis_data(self) -> str:
        """analysis.ymlの内容を文字列として取得."""
        if not self.analysis_file.exists():
            return ""
        
        try:
            with open(self.analysis_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

    def _generate_intention(self) -> dict[str, Any] | None:
        """LLMを使ってintentionを生成."""
        if not self.llm_model:
            return None

        # 必要なデータを取得
        latest_select_sentence = self._get_latest_select_sentence()
        latest_desire = self._get_latest_desire()
        analysis_data = self._get_analysis_data()
        status_data = self._load_yaml_file(self.status_file)
        my_log_data = self._load_yaml_file(self.my_log_file)

        # データが不足している場合はスキップ
        if not latest_select_sentence or not latest_desire:
            print(f"Insufficient data for intention generation for {self.agent_name}")
            return None

        try:
            # プロンプトテンプレートを取得
            prompt_template = self.config.get("prompt", {}).get("intention_generation", "")
            if not prompt_template:
                print("Intention generation prompt not found in config")
                return None

            # テンプレートを適用
            template = Template(prompt_template)
            prompt = template.render(
                agent_name=self.agent_name,
                analysis_data=analysis_data,
                latest_select_sentence_content=latest_select_sentence.get("content", ""),
                latest_select_sentence_request_count=latest_select_sentence.get("request_count", 0),
                latest_desire_content=latest_desire.get("desire", ""),
                latest_desire_request_count=latest_desire.get("request_count", 0),
                status_data=status_data,
                my_log_data=my_log_data,
            )

            # LLMに問い合わせ
            messages = [HumanMessage(content=prompt)]
            parser = StrOutputParser()
            chain = self.llm_model | parser
            response = chain.invoke(messages)

            # レスポンスを解析してconsistとcontentを抽出
            intention_data = self._parse_intention_response(response)
            
            if intention_data:
                # request_countを追加
                request_count = latest_desire.get("request_count", 0)
                intention_data["request_count"] = request_count
                
            return intention_data

        except Exception as e:
            print(f"Failed to generate intention: {e}")
            return None

    def _parse_intention_response(self, response: str) -> dict[str, Any] | None:
        """LLMレスポンスを解析してconsistとcontentを抽出."""
        try:
            lines = response.strip().split('\n')
            intention_data = {}
            current_key = None
            current_content = []

            for line in lines:
                line = line.strip()
                if line.startswith('consist:') or line.startswith('consist：'):
                    if current_key and current_content:
                        intention_data[current_key] = '\n'.join(current_content)
                    current_key = 'consist'
                    current_content = [line.split(':', 1)[-1].strip()]
                elif line.startswith('content:') or line.startswith('content：'):
                    if current_key and current_content:
                        intention_data[current_key] = '\n'.join(current_content)
                    current_key = 'content'
                    current_content = [line.split(':', 1)[-1].strip()]
                elif current_key and line:
                    current_content.append(line)

            # 最後のキーの内容を追加
            if current_key and current_content:
                intention_data[current_key] = '\n'.join(current_content)

            # consistとcontentの両方が存在することを確認
            if 'consist' in intention_data and 'content' in intention_data:
                return intention_data
            else:
                print("Failed to parse consist and content from LLM response")
                return None

        except Exception as e:
            print(f"Failed to parse intention response: {e}")
            return None

    def save_intention(self, intention_data: dict[str, Any]) -> None:
        """intentionをrequest_countごとにYAML形式で保存."""
        try:
            # 既存のデータを読み込み
            existing_data = {}
            if self.intention_file.exists():
                existing_data = self._load_yaml_file(self.intention_file) or {}
            
            # request_countをキーとして保存
            request_count = intention_data.get("request_count", 0)
            existing_data[request_count] = {
                "consist": intention_data.get("consist", ""),
                "content": intention_data.get("content", "")
            }
            
            # request_countの順序でソートして保存（キーを整数として比較）
            sorted_data = dict(sorted(existing_data.items(), key=lambda x: int(x[0])))
            
            with open(self.intention_file, "w", encoding="utf-8") as f:
                yaml.dump(sorted_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        except Exception as e:
            print(f"Failed to save intention: {e}")

    def process_intention(self) -> None:
        """メイン処理: 最新のrequest_countに対応するintentionを生成して保存."""
        # 最新のselect_sentenceを取得
        latest_select_sentence = self._get_latest_select_sentence()
        if not latest_select_sentence:
            print(f"No select_sentence data available for {self.agent_name}")
            return
        
        # select_sentenceのrequest_countを取得
        current_request_count = latest_select_sentence.get("request_count", 0)
        
        # 既存のintentionファイルをチェック
        existing_intentions = {}
        if self.intention_file.exists():
            existing_intentions = self._load_yaml_file(self.intention_file) or {}
        
        # 既にこのrequest_countのintentionが存在するかチェック
        if current_request_count in existing_intentions:
            print(f"Intention for request_count {current_request_count} already exists for {self.agent_name}")
            return
        
        # intentionを生成
        intention_data = self._generate_intention()
        
        if intention_data:
            # request_countを確実に設定
            intention_data["request_count"] = current_request_count
            self.save_intention(intention_data)
            print(f"Intention generated and saved for {self.agent_name}, request_count: {current_request_count}")
        else:
            print(f"Failed to generate intention for {self.agent_name}")