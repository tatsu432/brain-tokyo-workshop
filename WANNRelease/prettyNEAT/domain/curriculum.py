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
    time_steps_threshold: float = 750,
    time_steps_threshold_wins: float = 800,
) -> str:
    """Update curriculum stage based on population statistics.

    Args:
        pop_stats: Dictionary with 'avg_total_steps'
        current_stage: Current curriculum stage ('survival', 'mixed', 'wins', or None)
        time_steps_threshold: Average steps per episode needed to advance from 'survival' to 'mixed'
        time_steps_threshold_wins: Average steps per episode needed to advance from 'mixed' to 'wins'
                                   (maintains survival skill at higher level)

    Returns:
        Updated curriculum stage
    """
    if current_stage is None:
        return "survival"

    avg_total_steps = pop_stats.get("avg_total_steps", 0)

    if current_stage == "survival":
        # Advance from survival to mixed: check survival metric (average steps per episode)
        if avg_total_steps >= time_steps_threshold:
            logger.info(
                f"Curriculum: Advancing from 'survival' to 'mixed' (avg_total_steps={avg_total_steps:.1f})"
            )
            return "mixed"

    elif current_stage == "mixed":
        # Advance from mixed to wins: check survival metric at higher threshold
        if avg_total_steps >= time_steps_threshold_wins:
            logger.info(
                f"Curriculum: Advancing from 'mixed' to 'wins' "
                f"(avg_total_steps={avg_total_steps:.1f})"
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
