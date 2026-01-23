"""Self-play configuration."""
from dataclasses import dataclass


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    # Whether to enable self-play
    enabled: bool = False

    # Evaluation mode: 'baseline', 'archive', 'mixed'
    eval_mode: str = "mixed"

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
    touch_threshold: float = 5.0  # Avg touches to advance from 'touch' stage
    rally_threshold: float = 0.0  # Avg rally diff to advance from 'rally' stage

    @classmethod
    def from_hyperparameters(cls, hyp: dict) -> "SelfPlayConfig":
        """Create config from hyperparameters dictionary."""
        config = cls()
        config.enabled = hyp.get("selfplay_enabled", False)
        config.eval_mode = hyp.get("selfplay_eval_mode", "mixed")
        config.baseline_weight = hyp.get("selfplay_baseline_weight", 0.6)
        config.archive_weight = hyp.get("selfplay_archive_weight", 0.4)
        config.n_archive_opponents = hyp.get("selfplay_n_archive_opponents", 3)
        config.archive_add_frequency = hyp.get("selfplay_archive_add_freq", 5)
        config.enable_curriculum = hyp.get("selfplay_enable_curriculum", True)
        config.touch_threshold = hyp.get("touch_threshold", 5.0)
        config.rally_threshold = hyp.get("rally_threshold", 0.0)
        return config
