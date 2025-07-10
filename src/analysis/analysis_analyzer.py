"""Analysis analyzer module for processing talk messages and creating analysis.yml."""

from typing import Dict, List, Optional
from pathlib import Path
import logging
from aiwolf_nlp_common.packet import Packet, Request, Info, Talk
from .models import AnalysisData, AnalysisEntry, MessageType
from .llm_client import AnalysisLLMClient
from .dummy_llm_client import DummyLLMClient


class AnalysisAnalyzer:
    """Analyzer for processing talk messages and creating analysis entries."""
    
    def __init__(self, config: Dict, output_dir: Optional[Path] = None):
        """Initialize analysis analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save analysis files
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set output directory
        analysis_config = config.get("analysis", {})
        self.output_dir = output_dir or Path(analysis_config.get("output_dir", "./analysis_results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM analyzer
        try:
            self.llm_client = AnalysisLLMClient(config)
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client, using dummy client: {e}")
            self.llm_client = DummyLLMClient(config)
        
        # Analysis data per game
        self.game_analysis: Dict[str, AnalysisData] = {}  # game_id -> AnalysisData
        
        # Track all agent names per game
        self.game_agents: Dict[str, List[str]] = {}  # game_id -> list of agent names
    
    def process_initialize(self, packet: Packet, game_dir: Optional[Path] = None):
        """Process INITIALIZE packet to set up agent list.
        
        Args:
            packet: INITIALIZE packet
            game_dir: Optional game directory override
        """
        if packet.request != Request.INITIALIZE or not packet.info:
            return
        
        game_id = packet.info.game_id
        
        # Initialize game data structures
        if game_id not in self.game_analysis:
            self.game_analysis[game_id] = AnalysisData()
        
        # Store all agent names
        self.game_agents[game_id] = list(packet.info.status_map.keys())
    
    def process_talk(self, packet: Packet, game_dir: Optional[Path] = None):
        """Process TALK packet to analyze messages.
        
        Args:
            packet: TALK packet
            game_dir: Optional game directory override
        """
        if packet.request != Request.TALK or not packet.info or not packet.talk_history:
            return
        
        game_id = packet.info.game_id
        day = packet.info.day
        
        # Ensure we have analysis data for this game
        if game_id not in self.game_analysis:
            self.game_analysis[game_id] = AnalysisData()
        
        if game_id not in self.game_agents:
            # If we missed INITIALIZE, get agents from current status_map
            self.game_agents[game_id] = list(packet.info.status_map.keys())
        
        analysis_data = self.game_analysis[game_id]
        agent_names = self.game_agents[game_id]
        
        # Track talk_request_idx (number of TALK requests processed for this game)
        if not hasattr(self, 'game_talk_request_idx'):
            self.game_talk_request_idx = {}
        
        if game_id not in self.game_talk_request_idx:
            self.game_talk_request_idx[game_id] = 0
        
        # Increment talk request index for this game
        self.game_talk_request_idx[game_id] += 1
        current_talk_request_idx = self.game_talk_request_idx[game_id]
        
        # Process each talk entry
        for talk in packet.talk_history:
            from_agent = talk.agent
            text = talk.text
            idx = talk.idx
            
            # Skip if it's a skip or over message
            if talk.skip or talk.over or not text:
                continue
            
            # Check if text mentions agents or addresses the group
            has_mention, mentioned_agents, is_group_address = self.llm_client.check_agent_mention(
                text, agent_names
            )
            
            # Determine message type and create entry for ALL texts
            if is_group_address and not mentioned_agents:
                # Pure group address
                entry = AnalysisEntry(
                    type=MessageType.GROUP_ADDRESS,
                    from_agent=from_agent,
                    to_agents=[],  # Empty list for group address
                    topic=text,
                    day=day,
                    idx=idx,
                    talk_request_idx=current_talk_request_idx
                )
                analysis_data.add_entry(entry)
                self.logger.debug(f"Added group address entry: {from_agent} -> all")
            
            elif mentioned_agents:
                # Analyze message type for mentioned agents
                message_type_str = self.llm_client.analyze_message_type(text, mentioned_agents)
                
                # Convert string to MessageType enum
                try:
                    message_type = MessageType(message_type_str)
                except ValueError:
                    # Default to question if unknown type
                    message_type = MessageType.QUESTION
                
                entry = AnalysisEntry(
                    type=message_type,
                    from_agent=from_agent,
                    to_agents=mentioned_agents,
                    topic=text,
                    day=day,
                    idx=idx,
                    talk_request_idx=current_talk_request_idx
                )
                analysis_data.add_entry(entry)
                self.logger.debug(f"Added {message_type.value} entry: {from_agent} -> {mentioned_agents}")
            
            else:
                # No agent mention or group address - save with type NONE
                entry = AnalysisEntry(
                    type=MessageType.NONE,
                    from_agent=from_agent,
                    to_agents=[],  # Empty list since no specific target
                    topic=text,
                    day=day,
                    idx=idx,
                    talk_request_idx=current_talk_request_idx
                )
                analysis_data.add_entry(entry)
                self.logger.debug(f"Added none entry: {from_agent} (no specific target)")
    
    def save_analysis(self, game_id: str, game_dir: Optional[Path] = None):
        """Save analysis data for a specific game.
        
        Args:
            game_id: Game ID
            game_dir: Optional game directory override
        """
        if game_id not in self.game_analysis:
            return
        
        # Use provided directory or create default one
        if game_dir is None:
            game_dir = self.output_dir / game_id
        
        game_dir.mkdir(parents=True, exist_ok=True)
        
        # Save analysis
        analysis_file = game_dir / "analysis.yml"
        self.game_analysis[game_id].save(analysis_file)
        self.logger.info(f"Saved analysis for game {game_id} to {analysis_file}")
    
    def finalize_game(self, game_id: str, game_dir: Optional[Path] = None):
        """Finalize and save analysis data for a completed game.
        
        Args:
            game_id: Game ID
            game_dir: Optional game directory override
        """
        if game_id not in self.game_analysis:
            return
        
        # Save analysis
        self.save_analysis(game_id, game_dir)
        
        # Clean up game data
        del self.game_analysis[game_id]
        if game_id in self.game_agents:
            del self.game_agents[game_id]
        if hasattr(self, 'game_talk_request_idx') and game_id in self.game_talk_request_idx:
            del self.game_talk_request_idx[game_id]