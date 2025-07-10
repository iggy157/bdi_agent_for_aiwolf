"""Data models for analysis system."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any
import yaml
from pathlib import Path


class MessageType(str, Enum):
    """Types of messages in analysis."""
    QUESTION = "question"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    GROUP_ADDRESS = "group_address"
    NONE = "none"  # For messages without agent mentions or group addresses


@dataclass
class StatusEntry:
    """Status entry for each agent per day."""
    agent_name: str
    self_co: Optional[str] = None  # Self coming out role
    seer_co: Optional[str] = None  # Role reported by self-proclaimed seer
    alive: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "agent_name": self.agent_name,
            "self_co": self.self_co,
            "seer_co": self.seer_co,
            "alive": self.alive
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StatusEntry":
        """Create from dictionary."""
        return cls(
            agent_name=data["agent_name"],
            self_co=data.get("self_co"),
            seer_co=data.get("seer_co"),
            alive=data.get("alive", True)
        )


@dataclass
class AnalysisEntry:
    """Analysis entry for messages containing agent names or group addresses."""
    type: MessageType
    from_agent: str
    to_agents: List[str]
    topic: str
    day: int
    idx: int
    talk_request_idx: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "type": self.type.value,
            "from": self.from_agent,
            "to": self.to_agents,
            "topic": self.topic,
            "day": self.day,
            "idx": self.idx,
            "talk_request_idx": self.talk_request_idx
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisEntry":
        """Create from dictionary."""
        return cls(
            type=MessageType(data["type"]),
            from_agent=data["from"],
            to_agents=data["to"],
            topic=data["topic"],
            day=data["day"],
            idx=data["idx"],
            talk_request_idx=data.get("talk_request_idx", 0)  # Default to 0 for backwards compatibility
        )


class StatusData:
    """Manager for status data."""
    
    def __init__(self):
        self.entries: Dict[str, StatusEntry] = {}
    
    def update_agent(self, agent_name: str, **kwargs):
        """Update or create agent status."""
        if agent_name not in self.entries:
            self.entries[agent_name] = StatusEntry(agent_name=agent_name)
        
        for key, value in kwargs.items():
            if hasattr(self.entries[agent_name], key):
                setattr(self.entries[agent_name], key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            agent_name: entry.to_dict() 
            for agent_name, entry in self.entries.items()
        }
    
    def save(self, filepath: Path):
        """Save to YAML file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: Path) -> "StatusData":
        """Load from YAML file."""
        if not filepath.exists():
            return cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        
        status_data = cls()
        for agent_name, entry_data in data.items():
            entry_data["agent_name"] = agent_name
            status_data.entries[agent_name] = StatusEntry.from_dict(entry_data)
        
        return status_data


class AnalysisData:
    """Manager for analysis data."""
    
    def __init__(self):
        self.entries: List[AnalysisEntry] = []
    
    def add_entry(self, entry: AnalysisEntry):
        """Add analysis entry."""
        self.entries.append(entry)
    
    def to_dict(self) -> List[Dict[str, Any]]:
        """Convert to dictionary for YAML serialization."""
        return [entry.to_dict() for entry in self.entries]
    
    def save(self, filepath: Path):
        """Save to YAML file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: Path) -> "AnalysisData":
        """Load from YAML file."""
        if not filepath.exists():
            return cls()
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or []
        
        analysis_data = cls()
        for entry_data in data:
            analysis_data.entries.append(AnalysisEntry.from_dict(entry_data))
        
        return analysis_data