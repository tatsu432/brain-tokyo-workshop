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
    elite_fitness_threshold: float = 15.0,
    elite_fitness_threshold_wins: float = 15.7,
) -> str:
    """Update curriculum stage based on population statistics.

    Args:
        pop_stats: Dictionary with 'elite_fitness' (fitness of the best individual in the generation)
        current_stage: Current curriculum stage ('survival', 'mixed', 'wins', or None)
        elite_fitness_threshold: Elite fitness threshold needed to advance from 'survival' to 'mixed'
        elite_fitness_threshold_wins: Elite fitness threshold needed to advance from 'mixed' to 'wins'

    Returns:
        Updated curriculum stage
    """
    if current_stage is None:
        return "survival"

    elite_fitness = pop_stats.get("elite_fitness", 0.0)

    if current_stage == "survival":
        # Advance from survival to mixed: check elite fitness
        if elite_fitness >= elite_fitness_threshold:
            logger.info(
                f"Curriculum: Advancing from 'survival' to 'mixed' (elite_fitness={elite_fitness:.1f})"
            )
            return "mixed"

    elif current_stage == "mixed":
        # Advance from mixed to wins: check elite fitness at higher threshold
        if elite_fitness >= elite_fitness_threshold_wins:
            logger.info(
                f"Curriculum: Advancing from 'mixed' to 'wins' "
                f"(elite_fitness={elite_fitness:.1f})"
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
