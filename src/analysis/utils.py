"""Utility functions for analysis system."""

from datetime import datetime
from pathlib import Path
import re


def create_game_directory_name(game_id: str, start_time: datetime = None) -> str:
    """Create a timestamp-based directory name for a game.
    
    Args:
        game_id: The game ID (used for game summary but not directory name)
        start_time: Game start time (defaults to current time)
        
    Returns:
        Timestamp-based directory name (YYYYMMDD_HHMMSS)
    """
    if start_time is None:
        start_time = datetime.now()
    
    # Pure timestamp format: YYYYMMDD_HHMMSS
    return start_time.strftime("%Y%m%d_%H%M%S")


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to be filesystem safe.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Replace invalid characters with underscores
    return re.sub(r'[<>:"/\\|?*]', '_', filename)


def create_game_summary(game_info: dict) -> dict:
    """Create a summary of game information.
    
    Args:
        game_info: Dictionary containing game information
        
    Returns:
        Game summary dictionary
    """
    return {
        "game_id": game_info.get("game_id"),
        "start_time": game_info.get("start_time"),
        "end_time": game_info.get("end_time"),
        "total_days": game_info.get("total_days", 0),
        "agents": game_info.get("agents", []),
        "final_status": game_info.get("final_status", {}),
        "analysis_version": "1.0"
    }