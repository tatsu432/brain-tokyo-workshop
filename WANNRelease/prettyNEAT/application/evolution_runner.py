"""Main evolution loop orchestration."""

import logging

import numpy as np
from domain.curriculum import broadcast_curriculum_stage, update_curriculum
from domain.selfplay.archive import OpponentArchive
from domain.selfplay.config import SelfPlayConfig
from infrastructure.mpi_communication import (
    batch_mpi_eval,
    batch_mpi_eval_selfplay,
)
from neat_src.dataGatherer import DataGatherer
from neat_src.neat import Neat

from application.data_manager import gather_data

logger = logging.getLogger(__name__)


class EvolutionRunner:
    """Orchestrates the NEAT evolution process."""

    def __init__(
        self,
        hyp: dict,
        file_name: str,
        selfplay_config: SelfPlayConfig,
        opponent_archive: OpponentArchive,
        n_worker: int,
    ):
        self.hyp = hyp
        self.file_name = file_name
        self.selfplay_config = selfplay_config
        self.opponent_archive = opponent_archive
        self.n_worker = n_worker
        self.data = DataGatherer(file_name, hyp)
        self.neat = Neat(hyp)
        self.render_manager = None
        self.curriculum_stage = None

        # Initialize rendering manager if enabled
        self._setup_rendering()

        # Initialize curriculum tracking
        self._setup_curriculum()

    def _setup_rendering(self):
        """Initialize rendering manager if enabled."""
        render_enabled = self.hyp.get("render_enabled", False)
        if render_enabled:
            from domain.render_selfplay import RenderManager

            render_interval = self.hyp.get("render_interval", 10)
            render_fps = self.hyp.get("render_fps", 50)
            render_max_steps = self.hyp.get("render_max_steps", 3000)

            self.render_manager = RenderManager(
                render_interval=render_interval,
                render_fps=render_fps,
                max_steps=render_max_steps,
                enabled=True,
            )
            logger.info(
                f"Rendering ENABLED: every {render_interval} generations @ {render_fps} FPS"
            )

    def _setup_curriculum(self):
        """Initialize curriculum tracking."""
        enable_curriculum = self.hyp.get("enable_curriculum", False)
        if enable_curriculum:
            self.curriculum_stage = "survival"

        if self.selfplay_config.enabled:
            if not hasattr(self.data, "selfplay_stats"):
                self.data.selfplay_stats = {
                    "avg_touches": [],
                    "avg_rallies_won": [],
                    "avg_rallies_lost": [],
                    "curriculum_stage": [],
                    "archive_size": [],
                }
        elif enable_curriculum:
            if not hasattr(self.data, "curriculum_stats"):
                self.data.curriculum_stats = {
                    "avg_touches": [],
                    "avg_rallies_won": [],
                    "avg_rallies_lost": [],
                    "curriculum_stage": [],
                }

    def run(self):
        """Run the evolution loop."""
        verbose = self._is_verbose()

        # Set discrete action flag based on task config
        from domain import games

        task_config = games[self.hyp["task"]]
        self.data.is_discrete_action = task_config.actionSelect in ["prob", "hard"]

        # Configure display settings from hyperparameters
        display_config = self.hyp.get("display_config", {})
        if display_config:
            self.data.set_display_config(**display_config)

        if self.selfplay_config.enabled:
            logger.info("Self-play mode ENABLED")
        elif self.hyp.get("enable_curriculum", False):
            logger.info("Curriculum learning ENABLED (non-self-play)")

        for gen in range(self.hyp["maxGen"]):
            if verbose:
                logger.info(f"[Master] ===== Generation {gen} =====")

            pop = self.neat.ask()  # Get newly evolved individuals from NEAT

            # Evaluate population
            if self.selfplay_config.enabled:
                reward, action_dist, pop_stats, raw_reward = batch_mpi_eval_selfplay(
                    pop,
                    self.n_worker,
                    self.hyp,
                    track_actions=True,
                    verbose=verbose,
                )

                # Update curriculum based on population statistics
                if self.selfplay_config.enable_curriculum:
                    self.curriculum_stage = update_curriculum(
                        pop_stats,
                        self.curriculum_stage,
                        self.selfplay_config.touch_threshold,
                        self.selfplay_config.rally_threshold,
                    )
                    broadcast_curriculum_stage(self.curriculum_stage, self.n_worker)

                # Add best individual to archive periodically
                if gen % self.selfplay_config.archive_add_frequency == 0 and gen > 0:
                    self._maybe_add_to_archive(pop, reward, gen, verbose)

                # Track self-play stats
                self._track_selfplay_stats(pop_stats, gen)

            else:
                # Evaluate population for non-self-play
                reward, action_dist, pop_stats, raw_reward = batch_mpi_eval(
                    pop, self.n_worker, self.hyp, track_actions=True
                )

                # Update curriculum for non-self-play if enabled
                enable_curriculum = self.hyp.get("enable_curriculum", False)
                if enable_curriculum:
                    if self.curriculum_stage is None:
                        self.curriculum_stage = "survival"
                    self.curriculum_stage = update_curriculum(
                        pop_stats,
                        self.curriculum_stage,
                        self.hyp.get("touch_threshold", 5.0),
                        self.hyp.get("rally_threshold", 0.0),
                    )
                    broadcast_curriculum_stage(self.curriculum_stage, self.n_worker)

                # Track curriculum stats for non-self-play
                if enable_curriculum and hasattr(self.data, "curriculum_stats"):
                    self._track_curriculum_stats(pop_stats)

            self.neat.tell(reward)  # Send fitness to NEAT

            # Determine which eval function to use for check_best
            eval_fn = (
                batch_mpi_eval_selfplay
                if self.selfplay_config.enabled
                else batch_mpi_eval
            )

            self.data = gather_data(
                self.data,
                self.neat,
                gen,
                self.hyp,
                action_dist=action_dist,
                raw_fitness=raw_reward,
                shaped_stats=pop_stats,
                batch_eval_fn=eval_fn,
            )

            # Display with curriculum/self-play info
            self._log_generation(gen)

            # Trigger non-blocking rendering of best individual
            if self.render_manager is not None and self.render_manager.should_render(
                gen
            ):
                self._render_best(pop, reward, task_config, gen)

        # Clean up and data gathering at run end
        self._finalize()

    def _maybe_add_to_archive(self, pop, reward, gen, verbose):
        """Add best individual to archive if it meets criteria."""
        best_idx = np.argmax(reward)
        best_fitness = reward[best_idx]

        if verbose:
            logger.info(
                f"[Master] Gen {gen:4d}: Checking archive add (best_fitness={best_fitness:.2f}, threshold={self.selfplay_config.archive_fitness_threshold})",
            )

        if best_fitness > self.selfplay_config.archive_fitness_threshold:
            best_ind = pop[best_idx]
            w_vec_to_add = best_ind.wMat.flatten()
            a_vec_to_add = best_ind.aVec.flatten()

            if verbose:
                logger.info(
                    f"[Master] Adding to archive: wVec shape={w_vec_to_add.shape}, aVec shape={a_vec_to_add.shape}",
                )

            self.opponent_archive.add(w_vec_to_add, a_vec_to_add, best_fitness, gen)
            # Broadcast updated archive to workers
            self.opponent_archive.broadcast(self.n_worker)

    def _track_selfplay_stats(self, pop_stats, gen):
        """Track self-play statistics."""
        self.data.selfplay_stats["avg_touches"].append(pop_stats.get("avg_touches", 0))
        self.data.selfplay_stats["avg_rallies_won"].append(
            pop_stats.get("avg_rallies_won", 0)
        )
        self.data.selfplay_stats["avg_rallies_lost"].append(
            pop_stats.get("avg_rallies_lost", 0)
        )
        self.data.selfplay_stats["curriculum_stage"].append(self.curriculum_stage)
        self.data.selfplay_stats["archive_size"].append(
            len(self.opponent_archive.archive)
        )

    def _track_curriculum_stats(self, pop_stats):
        """Track curriculum statistics for non-self-play."""
        self.data.curriculum_stats["avg_touches"].append(
            pop_stats.get("avg_touches", 0)
        )
        self.data.curriculum_stats["avg_rallies_won"].append(
            pop_stats.get("avg_rallies_won", 0)
        )
        self.data.curriculum_stats["avg_rallies_lost"].append(
            pop_stats.get("avg_rallies_lost", 0)
        )
        if self.curriculum_stage is not None:
            self.data.curriculum_stats["curriculum_stage"].append(self.curriculum_stage)
        else:
            self.data.curriculum_stats["curriculum_stage"].append("survival")

    def _log_generation(self, gen):
        """Log generation information."""
        enable_curriculum = self.hyp.get("enable_curriculum", False)
        if self.selfplay_config.enabled:
            logger.info(
                f"Gen {gen:4d}: {self.data.display()} | Stage: {self.curriculum_stage}, Archive: {len(self.opponent_archive.archive)}"
            )
        elif enable_curriculum:
            if self.curriculum_stage is not None:
                logger.info(
                    f"Gen {gen:4d}: {self.data.display()} | Stage: {self.curriculum_stage}"
                )
            else:
                logger.info(f"Gen {gen:4d}: {self.data.display()}")
        else:
            logger.info(f"Gen {gen:4d}: {self.data.display()}")

    def _render_best(self, pop, reward, task_config, gen):
        """Render the best individual."""
        best_idx = np.argmax(reward)
        best_ind = pop[best_idx]

        # Update archive in render manager for self-play visualization
        if self.selfplay_config.enabled and self.opponent_archive.archive:
            self.render_manager.update_archive(self.opponent_archive.archive)

        # Start non-blocking render
        self.render_manager.render_best(
            wVec=best_ind.wMat.flatten(),
            aVec=best_ind.aVec.flatten(),
            nInput=task_config.input_size,
            nOutput=task_config.output_size,
            actSelect=task_config.actionSelect,
            generation=gen,
            use_random_archive_opponent=self.hyp.get(
                "render_use_archive_opponent", True
            )
            if self.selfplay_config.enabled
            else False,
        )

    def _finalize(self):
        """Finalize the evolution run."""
        gen = self.hyp["maxGen"] - 1
        eval_fn = (
            batch_mpi_eval_selfplay if self.selfplay_config.enabled else batch_mpi_eval
        )
        self.data = gather_data(
            self.data, self.neat, gen, self.hyp, save_pop=True, batch_eval_fn=eval_fn
        )
        self.data.save()
        self.data.savePop(self.neat.pop, self.file_name)

        # Save archive if self-play was used
        if self.selfplay_config.enabled and self.opponent_archive.archive:
            self.opponent_archive.save(self.file_name)

        # Clean up render manager
        if self.render_manager is not None:
            logger.debug("Waiting for render to finish...")
            self.render_manager.wait_for_render(timeout=10.0)
            self.render_manager.stop()

    def _is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        import os

        return os.environ.get("NEAT_VERBOSE", "0") == "1"
