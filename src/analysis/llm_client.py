"""LLM client factory for analysis using the same pattern as the main agent."""

import os
import json
from typing import Dict, Any, Optional, List, Tuple
from pydantic import SecretStr
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from jinja2 import Template
import logging


class AnalysisLLMClient:
    """LLM client for analysis that follows the same pattern as the main agent."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize LLM client based on config.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.llm_model = None
        self.prompts = config.get("analysis", {}).get("prompts", {})
        self.system_prompt = self.prompts.get("system", "あなたは人狼ゲームの発言を分析する専門家です。")
        
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM model based on config type."""
        model_type = str(self.config["llm"]["type"])
        
        try:
            match model_type:
                case "openai":
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        raise ValueError("OPENAI_API_KEY environment variable is not set")
                    self.llm_model = ChatOpenAI(
                        model=str(self.config["openai"]["model"]),
                        temperature=float(self.config["openai"]["temperature"]),
                        api_key=SecretStr(api_key),
                    )
                case "google":
                    api_key = os.environ.get("GOOGLE_API_KEY")
                    if not api_key:
                        raise ValueError("GOOGLE_API_KEY environment variable is not set")
                    self.llm_model = ChatGoogleGenerativeAI(
                        model=str(self.config["google"]["model"]),
                        temperature=float(self.config["google"]["temperature"]),
                        api_key=SecretStr(api_key),
                    )
                case "ollama":
                    self.llm_model = ChatOllama(
                        model=str(self.config["ollama"]["model"]),
                        temperature=float(self.config["ollama"]["temperature"]),
                        base_url=str(self.config["ollama"]["base_url"]),
                    )
                case _:
                    raise ValueError(f"Unknown LLM type: {model_type}")
            
            self.logger.info(f"Successfully initialized {model_type} LLM for analysis")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize {model_type} LLM for analysis: {e}")
            raise
    
    def _invoke_llm(self, prompt: str) -> Optional[str]:
        """Invoke LLM with prompt and return response.
        
        Args:
            prompt: The prompt to send to LLM
            
        Returns:
            LLM response text or None if error
        """
        try:
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm_model.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {e}")
            return None
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """Parse JSON response from LLM.
        
        Args:
            response: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary or None if parsing failed
        """
        if not response:
            return None
        
        try:
            # First try to parse the entire response as JSON
            return json.loads(response.strip())
        except json.JSONDecodeError:
            pass
        
        try:
            # Try to extract JSON from response (look for { ... })
            start = response.find('{')
            end = response.rfind('}') + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        try:
            # Try to extract JSON from code blocks (```json ... ```)
            import re
            json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            match = re.search(json_pattern, response, re.DOTALL)
            if match:
                return json.loads(match.group(1))
        except (json.JSONDecodeError, AttributeError):
            pass
        
        self.logger.error(f"Failed to parse JSON response: {response[:200]}...")
        return None
    
    def analyze_coming_out(self, text: str, agent_name: str) -> Optional[str]:
        """Analyze if text contains a coming out statement about role.
        
        Args:
            text: The message text to analyze
            agent_name: The name of the agent who sent the message
            
        Returns:
            Role name if coming out, None otherwise
        """
        template = Template(self.prompts.get("coming_out", ""))
        prompt = template.render(text=text, agent_name=agent_name)
        
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        
        if result and result.get("is_coming_out") and result.get("claimed_role"):
            return result["claimed_role"]
        
        return None
    
    def analyze_seer_report(self, text: str, reporter_name: str, target_agents: List[str]) -> Dict[str, Optional[str]]:
        """Analyze if text contains seer reports about other agents.
        
        Args:
            text: The message text to analyze
            reporter_name: The name of the self-proclaimed seer
            target_agents: List of agent names to check for reports
            
        Returns:
            Dictionary mapping agent names to reported roles (or None if not reported)
        """
        agents_str = ", ".join(target_agents)
        template = Template(self.prompts.get("seer_report", ""))
        prompt = template.render(text=text, reporter_name=reporter_name, agents_str=agents_str)
        
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        
        if result and "reports" in result:
            reports = result["reports"]
            return {agent: reports.get(agent) for agent in target_agents}
        
        return {agent: None for agent in target_agents}
    
    def check_agent_mention(self, text: str, agent_names: List[str]) -> Tuple[bool, List[str], bool]:
        """Check if text mentions specific agents or addresses the group.
        
        Args:
            text: The message text to analyze
            agent_names: List of all agent names in the game
            
        Returns:
            Tuple of (has_mention, mentioned_agents, is_group_address)
        """
        agents_str = ", ".join(agent_names)
        template = Template(self.prompts.get("agent_mention", ""))
        prompt = template.render(text=text, agents_str=agents_str)
        
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        
        if result:
            has_mention = result.get("has_agent_mention", False)
            mentioned_agents = result.get("mentioned_agents", [])
            is_group_address = result.get("is_group_address", False)
            
            # Filter to ensure only valid agent names are returned
            valid_agents = [agent for agent in mentioned_agents if agent in agent_names]
            
            return has_mention or is_group_address, valid_agents, is_group_address
        
        return False, [], False
    
    def analyze_message_type(self, text: str, to_agents: List[str]) -> str:
        """Analyze the type of message directed to specific agents.
        
        Args:
            text: The message text to analyze
            to_agents: List of agents the message is directed to
            
        Returns:
            Message type: "question", "positive", "negative", or "group_address"
        """
        if not to_agents:
            return "group_address"
        
        agents_str = ", ".join(to_agents)
        template = Template(self.prompts.get("message_type", ""))
        prompt = template.render(text=text, agents_str=agents_str)
        
        response = self._invoke_llm(prompt)
        result = self._parse_json_response(response)
        
        if result and "type" in result:
            return result["type"]
        
        return "question"  # Default fallback