"""Status analyzer module for processing game packets and updating status.yml."""

from typing import Dict, List, Optional
from pathlib import Path
import logging
from aiwolf_nlp_common.packet import Packet, Request, Info, Talk
from .models import StatusData
from .llm_client import AnalysisLLMClient
from .dummy_llm_client import DummyLLMClient


class StatusAnalyzer:
    """Analyzer for processing status information from game packets."""
    
    def __init__(self, config: Dict, output_dir: Optional[Path] = None):
        """Initialize status analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save status files
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
        
        # Status data per game
        self.game_status: Dict[str, Dict[int, StatusData]] = {}  # game_id -> day -> StatusData
        
        # Track self-proclaimed seers per game
        self.seer_agents: Dict[str, List[str]] = {}  # game_id -> list of self-proclaimed seer agents
    
    def process_initialize(self, packet: Packet, game_dir: Optional[Path] = None):
        """Process INITIALIZE packet to set up initial status.
        
        Args:
            packet: INITIALIZE packet
            game_dir: Optional game directory override
        """
        if packet.request != Request.INITIALIZE or not packet.info:
            return
        
        game_id = packet.info.game_id
        day = packet.info.day
        
        # Initialize game data structures
        if game_id not in self.game_status:
            self.game_status[game_id] = {}
            self.seer_agents[game_id] = []
        
        # Initialize day 0 status
        if day not in self.game_status[game_id]:
            self.game_status[game_id][day] = StatusData()
        
        # Add all agents to status with initial values
        for agent_name, status in packet.info.status_map.items():
            self.game_status[game_id][day].update_agent(
                agent_name=agent_name,
                alive=(status.value == "ALIVE")
            )
    
    def process_daily_initialize(self, packet: Packet, game_dir: Optional[Path] = None):
        """Process DAILY_INITIALIZE packet to update agent status.
        
        Args:
            packet: DAILY_INITIALIZE packet
            game_dir: Optional game directory override
        """
        if packet.request != Request.DAILY_INITIALIZE or not packet.info:
            return
        
        game_id = packet.info.game_id
        day = packet.info.day
        
        # Copy previous day's status as starting point
        if game_id in self.game_status and day > 0:
            prev_day = day - 1
            if prev_day in self.game_status[game_id]:
                # Create new StatusData for this day
                self.game_status[game_id][day] = StatusData()
                
                # Copy previous day's data
                for agent_name, prev_entry in self.game_status[game_id][prev_day].entries.items():
                    self.game_status[game_id][day].update_agent(
                        agent_name=agent_name,
                        self_co=prev_entry.self_co,
                        seer_co=prev_entry.seer_co,
                        alive=prev_entry.alive
                    )
        
        # Update alive status based on current status_map
        for agent_name, status in packet.info.status_map.items():
            self.game_status[game_id][day].update_agent(
                agent_name=agent_name,
                alive=(status.value == "ALIVE")
            )
    
    def process_talk(self, packet: Packet, game_dir: Optional[Path] = None):
        """Process TALK packet to analyze coming out statements.
        
        Args:
            packet: TALK packet
            game_dir: Optional game directory override
        """
        if packet.request != Request.TALK or not packet.info or not packet.talk_history:
            return
        
        game_id = packet.info.game_id
        day = packet.info.day
        
        # Ensure we have status data for this day
        if game_id not in self.game_status or day not in self.game_status[game_id]:
            return
        
        status_data = self.game_status[game_id][day]
        
        # Process each talk entry
        for talk in packet.talk_history:
            agent_name = talk.agent
            text = talk.text
            
            # Skip if it's a skip or over message
            if talk.skip or talk.over or not text:
                continue
            
            # Check for coming out statements
            claimed_role = self.llm_client.analyze_coming_out(text, agent_name)
            if claimed_role:
                # Update self CO
                status_data.update_agent(agent_name=agent_name, self_co=claimed_role)
                
                # Track if this agent claims to be a seer
                if claimed_role == "SEER" and agent_name not in self.seer_agents[game_id]:
                    self.seer_agents[game_id].append(agent_name)
                    self.logger.info(f"Agent {agent_name} claimed to be SEER in game {game_id}")
            
            # Check for seer reports (only from self-proclaimed seers)
            if agent_name in self.seer_agents.get(game_id, []):
                # Get list of other agents
                other_agents = [name for name in status_data.entries.keys() if name != agent_name]
                
                # Analyze seer reports
                reports = self.llm_client.analyze_seer_report(text, agent_name, other_agents)
                
                # Update seer CO for reported agents
                for target_agent, reported_role in reports.items():
                    if reported_role:
                        status_data.update_agent(agent_name=target_agent, seer_co=reported_role)
                        self.logger.info(f"Seer {agent_name} reported {target_agent} as {reported_role}")
    
    def save_status(self, game_id: str, day: int, game_dir: Optional[Path] = None):
        """Save status data for a specific game day.
        
        Args:
            game_id: Game ID
            day: Day number
            game_dir: Optional game directory override
        """
        if game_id not in self.game_status or day not in self.game_status[game_id]:
            return
        
        # Use provided directory or create default one
        if game_dir is None:
            game_dir = self.output_dir / game_id
        
        game_dir.mkdir(parents=True, exist_ok=True)
        
        # Save status for this day
        status_file = game_dir / f"status_day{day}.yml"
        self.game_status[game_id][day].save(status_file)
        self.logger.info(f"Saved status for game {game_id}, day {day} to {status_file}")
    
    def finalize_game(self, game_id: str, game_dir: Optional[Path] = None):
        """Finalize and save all status data for a completed game.
        
        Args:
            game_id: Game ID
            game_dir: Optional game directory override
        """
        if game_id not in self.game_status:
            return
        
        # Save all days' status
        for day in sorted(self.game_status[game_id].keys()):
            self.save_status(game_id, day, game_dir)
        
        # Clean up game data
        del self.game_status[game_id]
        if game_id in self.seer_agents:
            del self.seer_agents[game_id]