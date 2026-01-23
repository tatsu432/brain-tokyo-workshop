"""NEAT training script with clean architecture.

This script orchestrates the NEAT evolution process using a clean architecture
with proper separation of concerns:
- Core: Environment setup and logging
- Domain: Self-play, curriculum learning
- Infrastructure: MPI communication
- Application: Evolution orchestration and data management
"""
import os
import sys
import argparse
import numpy as np
from mpi4py import MPI

# Setup environment and logging first
from core.setup import setup_environment, setup_logging, suppress_stderr

setup_environment()
logger = setup_logging()

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# NumPy settings
np.set_printoptions(precision=2, linewidth=160)

# Import NEAT and domain modules
from neat_src import loadHyp, updateHyp

with suppress_stderr():
    from domain import games  # Task environments

# Import application and infrastructure modules
from domain.selfplay.config import SelfPlayConfig
from domain.selfplay.archive import OpponentArchive
from application.evolution_runner import EvolutionRunner
from infrastructure.mpi_communication import run_worker, stop_all_workers, mpi_fork


def main(args):
    """Handles command line input, launches optimization or evaluation script
    depending on MPI rank.
    """
    # Load hyperparameters
    hyp_default = args.default
    hyp_adjust = args.hyperparam

    hyp = loadHyp(pFileName=hyp_default)
    updateHyp(hyp, hyp_adjust)

    # Apply command-line rendering overrides (only affects master)
    if args.render:
        hyp["render_enabled"] = True
    if args.no_render:
        hyp["render_enabled"] = False
    if args.render_interval is not None:
        hyp["render_interval"] = args.render_interval
        if args.render_interval > 0:
            hyp["render_enabled"] = True
    if args.render_fps is not None:
        hyp["render_fps"] = args.render_fps

    # Store nWorker in hyp for use by other modules
    n_worker = args.num_worker + 1  # +1 for master
    hyp["nWorker"] = n_worker

    # Launch main thread and workers
    if rank == 0:
        run_master(args.outPrefix, hyp, n_worker)
    else:
        run_worker(hyp)

    # Cleanup
    if rank == 0:
        stop_all_workers(n_worker)


def run_master(file_name: str, hyp: dict, n_worker: int):
    """Run the master process (evolution loop)."""
    # Initialize self-play configuration
    selfplay_config = SelfPlayConfig.from_hyperparameters(hyp)
    
    # Override with global enable_curriculum if set
    enable_curriculum = hyp.get("enable_curriculum", False)
    if enable_curriculum is not None:
        selfplay_config.enable_curriculum = enable_curriculum

    # Initialize opponent archive
    opponent_archive = OpponentArchive(selfplay_config)
    
    # Try to load existing archive if self-play is enabled
    if selfplay_config.enabled:
        opponent_archive.load(file_name)

    # Create and run evolution
    runner = EvolutionRunner(
        hyp=hyp,
        file_name=file_name,
        selfplay_config=selfplay_config,
        opponent_archive=opponent_archive,
        n_worker=n_worker,
    )
    runner.run()


if __name__ == "__main__":
    """Parse input and launch"""
    parser = argparse.ArgumentParser(description=("Evolve NEAT networks"))

    parser.add_argument(
        "-d",
        "--default",
        type=str,
        help="default hyperparameter file",
        default="p/default_neat.json",
    )

    parser.add_argument(
        "-p", "--hyperparam", type=str, help="hyperparameter file", default=None
    )

    parser.add_argument(
        "-o",
        "--outPrefix",
        type=str,
        help="file name for result output",
        default="test",
    )

    parser.add_argument(
        "-n", "--num_worker", type=int, help="number of cores to use", default=8
    )

    # Rendering options (override config file)
    parser.add_argument(
        "--render", action="store_true", help="enable rendering during training"
    )

    parser.add_argument(
        "--no-render", action="store_true", help="disable rendering during training"
    )

    parser.add_argument(
        "--render-interval",
        type=int,
        default=None,
        help="render every N generations (e.g., 10)",
    )

    parser.add_argument(
        "--render-fps", type=int, default=None, help="rendering frames per second"
    )

    args = parser.parse_args()

    # Use MPI if parallel
    if "parent" == mpi_fork(args.num_worker + 1):
        os._exit(0)

    main(args)
