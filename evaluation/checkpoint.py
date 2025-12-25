"""
Checkpoint Management for Evaluation Worker
Tracks last processed time to avoid re-evaluating spans
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional


# Configuration
DEFAULT_LOOKBACK_MINUTES = 10
CHECKPOINT_FILE = "worker_checkpoint.txt"


def load_checkpoint() -> datetime:
    """
    Load last processed time from checkpoint file
    
    Returns:
        datetime: Last processed time (UTC), or lookback time if no checkpoint
    """
    if not os.path.exists(CHECKPOINT_FILE):
        # First run → look back a bit
        fallback_time = datetime.now(timezone.utc) - timedelta(minutes=DEFAULT_LOOKBACK_MINUTES)
        print(f"[Checkpoint] No checkpoint file found, using {DEFAULT_LOOKBACK_MINUTES}min lookback")
        return fallback_time
    
    try:
        with open(CHECKPOINT_FILE, "r") as f:
            iso_time = f.read().strip()
            checkpoint_time = datetime.fromisoformat(iso_time)
            print(f"[Checkpoint] Loaded: {checkpoint_time.isoformat()}")
            return checkpoint_time
            
    except Exception as e:
        print(f"[Checkpoint] Warning: Could not load checkpoint: {e}")
        # Fallback if file is corrupted
        fallback_time = datetime.now(timezone.utc) - timedelta(minutes=DEFAULT_LOOKBACK_MINUTES)
        return fallback_time


def save_checkpoint(dt: datetime) -> None:
    """
    Save last processed time to checkpoint file
    
    Args:
        dt: Datetime to save (should be UTC)
    """
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            f.write(dt.isoformat())
        print(f"[Checkpoint] Saved: {dt.isoformat()}")
        
    except Exception as e:
        print(f"[Checkpoint] Warning: Could not save checkpoint: {e}")


def get_lookback_time(minutes: Optional[int] = None) -> datetime:
    """
    Get a datetime for looking back from now
    
    Args:
        minutes: Minutes to look back (default: DEFAULT_LOOKBACK_MINUTES)
    
    Returns:
        datetime: UTC time `minutes` ago
    """
    if minutes is None:
        minutes = DEFAULT_LOOKBACK_MINUTES
    
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)