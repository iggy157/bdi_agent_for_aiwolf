import os
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from jinja2 import Template
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class DesireTracker:
    """
    best_policyからLLMを使ってdesire文を生成し、保存するクラス
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定情報
        """
        self.config = config
        self.base_path = Path("info/desire")
        self.llm_model = None
        
        # .envファイルを読み込み
        load_dotenv(Path(__file__).parent.parent.parent / "config" / ".env")
        
        # LLMモデルを初期化
        self._initialize_llm()
    
    def _initialize_llm(self) -> None:
        """LLMモデルを初期化する"""
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
                    api_key=SecretStr(os.environ["GOOGLE_API_KEY"]),
                )
            else:
                raise ValueError(f"Unsupported LLM type: {model_type}")
                
        except Exception as e:
            print(f"LLM初期化エラー: {e}")
            self.llm_model = None
    
    def generate_desire_from_best_policy(self, best_policy: List[Dict[str, Any]], 
                                       game_id: str, agent_name: str, 
                                       info: Optional[Any] = None) -> str:
        """
        best_policyからLLMを使ってdesire文を生成
        
        Args:
            best_policy: 条件に一致しなかったpolicyルールのリスト
            game_id: ゲームID
            agent_name: エージェント名
            info: ゲーム情報（オプション）
            
        Returns:
            生成されたdesire文
        """
        if not best_policy:
            return "現在特に強い願望はありません。"
        
        # best_policyを文字列に整形
        policy_text = self._format_best_policy(best_policy)
        
        # LLMプロンプトを作成
        prompt = self._create_desire_prompt(policy_text, agent_name, info)
        
        # LLMでdesire文を生成
        try:            
            # LLMでdesire文を生成
            if self.llm_model:
                try:
                    response = (self.llm_model | StrOutputParser()).invoke([HumanMessage(content=prompt)])
                    return response.strip()
                except Exception as llm_error:
                    print(f"LLM呼び出しエラー: {llm_error}")
                    return self._create_fallback_desire(best_policy)
            else:
                print("LLMモデルが初期化されていません")
                return self._create_fallback_desire(best_policy)
                
        except Exception as e:
            print(f"Desire生成エラー: {e}")
            return self._create_fallback_desire(best_policy)
    
    def _format_best_policy(self, best_policy: List[Dict[str, Any]]) -> str:
        """
        best_policyを読みやすい形式に整形
        
        Args:
            best_policy: policyルールのリスト
            
        Returns:
            整形されたテキスト
        """
        formatted_parts = []
        
        for policy in sorted(best_policy, key=lambda x: x.get("priority", 0)):
            part = f"""
ルール{policy.get('id', 'N/A')}: {policy.get('name', '不明')}
条件: {policy.get('condition', '不明')}
行動: {policy.get('then', '不明')}
補足: {policy.get('supplement', '不明')}
優先度: {policy.get('priority', 0)}
"""
            formatted_parts.append(part.strip())
        
        return "\n\n".join(formatted_parts)
    
    def _create_desire_prompt(self, policy_text: str, agent_name: str, 
                            info: Optional[Any] = None) -> str:
        """
        LLM用のdesire生成プロンプトを作成
        
        Args:
            policy_text: 整形されたpolicy情報
            agent_name: エージェント名
            info: ゲーム情報
            
        Returns:
            LLMプロンプト
        """
        # generate_desire_from_policyプロンプトを取得
        template_str = self.config.get("prompt", {}).get("generate_desire_from_policy", 
            "{{ policy_rules }}")
        
        # Jinja2テンプレートでプロンプトを生成
        template = Template(template_str)
        policy_prompt = template.render(policy_rules=policy_text)
        
        # generate_desireプロンプトがある場合は追加
        desire_prompt = self.config.get("prompt", {}).get("generate_desire", "")
        
        context = ""
        if info:
            context = f"""
現在のゲーム状況:
- ゲーム日: {getattr(info, 'day', 'N/A')}
- 時間: {getattr(info, 'phase', 'N/A')}
- エージェント名: {agent_name}
"""
        
        # 完全なプロンプトを構築
        full_prompt = f"""現在実行できていないポリシールール:
{policy_prompt}

{context}

{desire_prompt if desire_prompt else '上記のルールを踏まえて、このエージェントが現在抱いている願望や意図を自然な日本語で表現してください。'}"""
        
        return full_prompt.strip()
    
    def _create_fallback_desire(self, best_policy: List[Dict[str, Any]]) -> str:
        """
        LLM生成に失敗した場合のフォールバック
        
        Args:
            best_policy: policyルールのリスト
            
        Returns:
            フォールバック用のdesire文
        """
        if not best_policy:
            return "現在特に強い願望はありません。"
        
        # 優先度が最も高いルールを基にフォールバック
        highest_priority = min(best_policy, key=lambda x: x.get("priority", 0))
        then_action = highest_priority.get("then", "適切な行動を取りたい")
        
        return f"現在の状況に応じて{then_action[:50]}..."
    
    def save_desire(self, desire_text: str, game_id: str, agent_name: str, 
                   info: Optional[Any] = None) -> None:
        """
        生成されたdesire文をファイルに保存
        
        Args:
            desire_text: 生成されたdesire文
            game_id: ゲームID
            agent_name: エージェント名
            info: ゲーム情報
        """
        # 保存先ディレクトリを作成
        save_dir = self.base_path / game_id / agent_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # desire.ymlファイルのパス
        desire_file = save_dir / "desire.yml"
        
        # 既存のdesireエントリを読み込み
        existing_desires = []
        if desire_file.exists():
            try:
                with open(desire_file, 'r', encoding='utf-8') as f:
                    existing_data = yaml.safe_load(f) or {}
                    existing_desires = existing_data.get("desires", [])
            except Exception:
                existing_desires = []
        
        # 新しいdesireエントリを作成
        new_entry = {
            "timestamp": datetime.now().isoformat(),
            "request_count": len(existing_desires) + 1,
            "desire": desire_text,
            "day": getattr(info, 'day', None) if info else None,
            "phase": getattr(info, 'phase', None) if info else None
        }
        
        # desireエントリを追加
        existing_desires.append(new_entry)
        
        # ファイルに保存
        output_data = {
            "agent_name": agent_name,
            "game_id": game_id,
            "total_desires": len(existing_desires),
            "desires": existing_desires
        }
        
        with open(desire_file, 'w', encoding='utf-8') as f:
            yaml.dump(output_data, f, default_flow_style=False, 
                     allow_unicode=True, indent=2)
    
    def generate_and_save_desire(self, best_policy: List[Dict[str, Any]], 
                               game_id: str, agent_name: str, 
                               info: Optional[Any] = None) -> str:
        """
        best_policyからdesire文を生成し、保存する統合メソッド
        
        Args:
            best_policy: 条件に一致しなかったpolicyルールのリスト
            game_id: ゲームID
            agent_name: エージェント名
            info: ゲーム情報
            
        Returns:
            生成されたdesire文
        """
        # desire文を生成
        desire_text = self.generate_desire_from_best_policy(
            best_policy, game_id, agent_name, info
        )
        
        # ファイルに保存
        self.save_desire(desire_text, game_id, agent_name, info)
        
        return desire_text
