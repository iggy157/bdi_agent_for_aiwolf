"""Main packet analyzer module that coordinates status and analysis processing."""

from typing import Dict, Optional
from pathlib import Path
import logging
from datetime import datetime
from aiwolf_nlp_common.packet import Packet, Request
from .status_analyzer import StatusAnalyzer
from .analysis_analyzer import AnalysisAnalyzer
from .utils import create_game_directory_name, create_game_summary
import yaml


class PacketAnalyzer:
    """Main analyzer that processes packets and coordinates sub-analyzers."""
    
    def __init__(self, config: Dict, output_dir: Optional[Path] = None):
        """Initialize packet analyzer.
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save analysis results
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.fail_safe = config.get("analysis", {}).get("fail_safe", False)
        
        # Initialize sub-analyzers
        try:
            self.status_analyzer = StatusAnalyzer(config, output_dir)
            self.analysis_analyzer = AnalysisAnalyzer(config, output_dir)
            self.logger.info("Analysis sub-systems initialized successfully")
        except Exception as e:
            if self.fail_safe:
                self.logger.warning(f"Analysis initialization failed, running in fail-safe mode: {e}")
                self.status_analyzer = None
                self.analysis_analyzer = None
            else:
                raise
        
        # Track active games and their information
        self.active_games = set()
        self.game_info: Dict[str, Dict] = {}  # game_id -> game_info
        self.game_directories: Dict[str, Path] = {}  # game_id -> actual directory path
    
    def process_packet(self, packet: Packet):
        """Process a single packet through all analyzers.
        
        Args:
            packet: The packet to process
        """
        if not packet.info:
            return
        
        # If running in fail-safe mode without analyzers, just log
        if self.fail_safe and not self.status_analyzer and not self.analysis_analyzer:
            self.logger.debug(f"Skipping packet processing in fail-safe mode: {packet.request}")
            return
        
        game_id = packet.info.game_id
        
        # Track active games
        if packet.request == Request.INITIALIZE:
            self.active_games.add(game_id)
            
            # Initialize game info
            start_time = datetime.now()
            self.game_info[game_id] = {
                "game_id": game_id,
                "start_time": start_time.isoformat(),
                "agents": list(packet.info.status_map.keys()) if packet.info.status_map else [],
                "total_days": 0,
                "final_status": {}
            }
            
            # Create game directory with descriptive name
            dir_name = create_game_directory_name(game_id, start_time)
            game_dir = self.config.get("analysis", {}).get("output_dir", "./analysis_results")
            game_dir = Path(game_dir) / dir_name
            game_dir.mkdir(parents=True, exist_ok=True)
            self.game_directories[game_id] = game_dir
            
            self.logger.info(f"Started tracking game: {game_id} -> {dir_name}")
        
        # Get game directory for this game
        game_dir = self.game_directories.get(game_id)
        
        # Process packet in each analyzer based on request type
        if packet.request == Request.INITIALIZE:
            if self.status_analyzer:
                self.status_analyzer.process_initialize(packet, game_dir)
            if self.analysis_analyzer:
                self.analysis_analyzer.process_initialize(packet, game_dir)
        
        elif packet.request == Request.DAILY_INITIALIZE:
            if self.status_analyzer:
                self.status_analyzer.process_daily_initialize(packet, game_dir)
            # Update total days
            if game_id in self.game_info:
                self.game_info[game_id]["total_days"] = max(
                    self.game_info[game_id]["total_days"], packet.info.day
                )
        
        elif packet.request == Request.TALK:
            if self.status_analyzer:
                self.status_analyzer.process_talk(packet, game_dir)
            if self.analysis_analyzer:
                self.analysis_analyzer.process_talk(packet, game_dir)
        
        elif packet.request == Request.DAILY_FINISH:
            # Save status at end of each day
            if self.status_analyzer:
                day = packet.info.day
                self.status_analyzer.save_status(game_id, day, game_dir)
        
        elif packet.request == Request.FINISH:
            # Update final status
            if game_id in self.game_info and packet.info.status_map:
                self.game_info[game_id]["final_status"] = {
                    agent: status.value for agent, status in packet.info.status_map.items()
                }
                self.game_info[game_id]["end_time"] = datetime.now().isoformat()
            
            # Finalize game
            self.finalize_game(game_id)
    
    def finalize_game(self, game_id: str):
        """Finalize all analysis for a completed game.
        
        Args:
            game_id: Game ID to finalize
        """
        if game_id not in self.active_games:
            return
        
        self.logger.info(f"Finalizing analysis for game: {game_id}")
        
        # Get game directory
        game_dir = self.game_directories.get(game_id)
        
        # Finalize in each analyzer
        if self.status_analyzer:
            self.status_analyzer.finalize_game(game_id, game_dir)
        if self.analysis_analyzer:
            self.analysis_analyzer.finalize_game(game_id, game_dir)
        
        # Save game summary
        if game_id in self.game_info and game_dir:
            summary = create_game_summary(self.game_info[game_id])
            summary_file = game_dir / "game_summary.yml"
            with open(summary_file, 'w', encoding='utf-8') as f:
                yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
            self.logger.info(f"Saved game summary to {summary_file}")
        
        # Cleanup
        self.active_games.remove(game_id)
        if game_id in self.game_info:
            del self.game_info[game_id]
        if game_id in self.game_directories:
            del self.game_directories[game_id]
        
        self.logger.info(f"Completed analysis for game: {game_id}")
    
    def shutdown(self):
        """Shutdown analyzer and finalize all active games."""
        self.logger.info("Shutting down packet analyzer")
        
        # Finalize all active games
        for game_id in list(self.active_games):
            self.finalize_game(game_id)