"""Curriculum learning management."""
import logging
from typing import Optional, Dict
from mpi4py import MPI

comm = MPI.COMM_WORLD

logger = logging.getLogger(__name__)

# Global storage for curriculum broadcasts
_pending_curriculum_data = []
_pending_curriculum_requests = []


def update_curriculum(
    pop_stats: Dict[str, float],
    current_stage: Optional[str],
    touch_threshold: float = 5.0,
    rally_threshold: float = 0.0,
) -> str:
    """Update curriculum stage based on population statistics.

    Args:
        pop_stats: Dictionary with 'avg_touches', 'avg_rallies_won', 'avg_rallies_lost'
        current_stage: Current curriculum stage ('survival', 'mixed', 'wins', or None)
        touch_threshold: Average touches needed to advance from 'survival' to 'mixed'
        rally_threshold: Average rally difference needed to advance from 'mixed' to 'wins'

    Returns:
        Updated curriculum stage
    """
    if current_stage is None:
        return "survival"

    avg_touches = pop_stats.get("avg_touches", 0)
    avg_rally_diff = pop_stats.get("avg_rallies_won", 0) - pop_stats.get(
        "avg_rallies_lost", 0
    )

    if current_stage == "survival":
        if avg_touches >= touch_threshold:
            logger.info(
                f"Curriculum: Advancing from 'survival' to 'mixed' (avg_touches={avg_touches:.1f})"
            )
            return "mixed"

    elif current_stage == "mixed":
        if avg_rally_diff >= rally_threshold:
            logger.info(
                f"Curriculum: Advancing from 'mixed' to 'wins' (rally_diff={avg_rally_diff:.1f})"
            )
            return "wins"

    return current_stage


def broadcast_curriculum_stage(stage: str, n_worker: int):
    """Broadcast curriculum stage to all workers using non-blocking sends."""
    global _pending_curriculum_data, _pending_curriculum_requests
    n_slave = n_worker - 1

    # Clear previous
    _pending_curriculum_data = []
    _pending_curriculum_requests = []

    # Use non-blocking sends with persistent references
    for i_work in range(1, n_slave + 1):
        msg = ("curriculum_update", stage)
        _pending_curriculum_data.append(msg)
        req = comm.isend(msg, dest=i_work, tag=101)
        _pending_curriculum_requests.append(req)
