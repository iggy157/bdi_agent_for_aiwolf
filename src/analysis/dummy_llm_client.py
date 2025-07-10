"""Dummy LLM client for testing analysis without actual LLM calls."""

from typing import Dict, List, Optional, Tuple
import logging
import random


class DummyLLMClient:
    """Dummy LLM client that provides fake analysis results for testing."""
    
    def __init__(self, config: Dict):
        """Initialize dummy client."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using dummy LLM client for analysis (no actual LLM calls)")
    
    def analyze_coming_out(self, text: str, agent_name: str) -> Optional[str]:
        """Dummy coming out analysis."""
        # Dummy logic: detect simple CO keywords
        text_lower = text.lower()
        roles = ["seer", "bodyguard", "medium", "villager", "werewolf", "possessed"]
        
        for role in roles:
            if role in text_lower or f"{role}" in text_lower:
                self.logger.debug(f"Dummy analysis: {agent_name} claims {role.upper()}")
                return role.upper()
        
        return None
    
    def analyze_seer_report(self, text: str, reporter_name: str, target_agents: List[str]) -> Dict[str, Optional[str]]:
        """Dummy seer report analysis."""
        result = {}
        text_lower = text.lower()
        
        for agent in target_agents:
            if agent.lower() in text_lower:
                # Simple keyword detection
                if any(word in text_lower for word in ["human", "white", "villager", "innocent"]):
                    result[agent] = "HUMAN"
                    self.logger.debug(f"Dummy analysis: {reporter_name} reports {agent} as HUMAN")
                elif any(word in text_lower for word in ["werewolf", "black", "wolf", "guilty"]):
                    result[agent] = "WEREWOLF"
                    self.logger.debug(f"Dummy analysis: {reporter_name} reports {agent} as WEREWOLF")
                else:
                    result[agent] = None
            else:
                result[agent] = None
        
        return result
    
    def check_agent_mention(self, text: str, agent_names: List[str]) -> Tuple[bool, List[str], bool]:
        """Dummy agent mention detection."""
        mentioned_agents = []
        text_lower = text.lower()
        
        # Check for agent mentions
        for agent in agent_names:
            if agent.lower() in text_lower:
                mentioned_agents.append(agent)
        
        # Check for group addresses
        group_words = ["everyone", "みんな", "all", "everybody", "皆", "全員"]
        is_group_address = any(word in text_lower for word in group_words)
        
        has_mention = len(mentioned_agents) > 0 or is_group_address
        
        if mentioned_agents:
            self.logger.debug(f"Dummy analysis: mentions {mentioned_agents}")
        if is_group_address:
            self.logger.debug("Dummy analysis: group address detected")
        
        return has_mention, mentioned_agents, is_group_address
    
    def analyze_message_type(self, text: str, to_agents: List[str]) -> str:
        """Dummy message type analysis."""
        text_lower = text.lower()
        
        # Simple keyword-based classification
        question_words = ["?", "？", "what", "who", "how", "why", "where", "when", "どう", "なぜ", "誰", "何"]
        positive_words = ["trust", "good", "believe", "agree", "信頼", "良い", "賛成", "いい"]
        negative_words = ["doubt", "suspicious", "disagree", "bad", "疑", "怪しい", "反対", "悪い"]
        
        if any(word in text_lower for word in question_words):
            self.logger.debug("Dummy analysis: classified as question")
            return "question"
        elif any(word in text_lower for word in positive_words):
            self.logger.debug("Dummy analysis: classified as positive")
            return "positive"
        elif any(word in text_lower for word in negative_words):
            self.logger.debug("Dummy analysis: classified as negative")
            return "negative"
        else:
            # Random fallback for testing
            choice = random.choice(["question", "positive", "negative"])
            self.logger.debug(f"Dummy analysis: random classification as {choice}")
            return choice