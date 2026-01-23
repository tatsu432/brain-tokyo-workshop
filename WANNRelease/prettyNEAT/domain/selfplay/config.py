"""Self-play configuration."""

from dataclasses import dataclass


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    # Whether to enable self-play
    enabled: bool = False

    # Evaluation mode: 'baseline', 'archive', 'mixed'
    eval_mode: str = "survival"

    # Weight for baseline vs archive evaluation
    baseline_weight: float = 0.6
    archive_weight: float = 0.4

    # Number of archive opponents to sample
    n_archive_opponents: int = 3

    # Archive settings
    max_archive_size: int = 50
    archive_add_frequency: int = 5  # Add best every N generations
    archive_fitness_threshold: float = -3.0  # Min fitness to add to archive

    # Curriculum settings
    enable_curriculum: bool = True
    time_steps_threshold: float = (
        750  # Avg steps per episode to advance from 'survival' to 'mixed' stage
    )
    time_steps_threshold_wins: float = (
        800  # Avg steps per episode to advance from 'mixed' to 'wins' stage
    )

    @classmethod
    def from_hyperparameters(cls, hyp: dict) -> "SelfPlayConfig":
        """Create config from hyperparameters dictionary."""
        config = cls()
        config.enabled = hyp.get("selfplay_enabled", False)
        config.eval_mode = hyp.get("selfplay_eval_mode", "survival")
        config.baseline_weight = hyp.get("selfplay_baseline_weight", 0.6)
        config.archive_weight = hyp.get("selfplay_archive_weight", 0.4)
        config.n_archive_opponents = hyp.get("selfplay_n_archive_opponents", 3)
        config.archive_add_frequency = hyp.get("selfplay_archive_add_freq", 5)
        config.enable_curriculum = hyp.get("selfplay_enable_curriculum", True)
        config.time_steps_threshold = hyp.get("time_steps_threshold", 750)
        config.time_steps_threshold_wins = hyp.get("time_steps_threshold_wins", 800)
        return config
