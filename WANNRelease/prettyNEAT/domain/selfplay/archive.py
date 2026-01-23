"""Opponent archive management for self-play."""
import os
import pickle
import numpy as np
from typing import List, Tuple
from mpi4py import MPI

from domain.selfplay.config import SelfPlayConfig

comm = MPI.COMM_WORLD


class OpponentArchive:
    """Manages the archive of opponent individuals for self-play."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.archive: List[Tuple[np.ndarray, np.ndarray, float, int]] = []
        # List of (wVec, aVec, fitness, generation) tuples

        # Global storage to keep non-blocking send requests and data alive
        self._pending_broadcast_data = []
        self._pending_broadcast_requests = []

    def add(self, w_vec: np.ndarray, a_vec: np.ndarray, fitness: float, generation: int):
        """Add an individual to the opponent archive."""
        if len(self.archive) < self.config.max_archive_size:
            self.archive.append((w_vec.copy(), a_vec.copy(), fitness, generation))
        else:
            # Replace worst if new individual is better
            worst_idx = min(
                range(len(self.archive)), key=lambda i: self.archive[i][2]
            )
            if fitness > self.archive[worst_idx][2]:
                self.archive[worst_idx] = (
                    w_vec.copy(),
                    a_vec.copy(),
                    fitness,
                    generation,
                )

    def broadcast(self, n_worker: int):
        """Broadcast archive to all workers using non-blocking sends.

        NOTE: We use isend() (non-blocking) because workers are blocked waiting for
        work on tag=1, not actively receiving tag=100. Blocking send could deadlock.

        IMPORTANT: We store the data and requests globally to prevent garbage collection
        before the sends complete. Workers receive these messages asynchronously.
        """
        n_slave = n_worker - 1
        verbose = os.environ.get("NEAT_VERBOSE", "0") == "1"

        if verbose:
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"[Master] Broadcasting archive (size={len(self.archive)}) to {n_slave} workers...",
            )

        # Clear any completed previous broadcasts
        self._pending_broadcast_data = []
        self._pending_broadcast_requests = []

        # Create a serializable snapshot of the archive
        archive_snapshot = [(w.copy(), a.copy(), f, g) for w, a, f, g in self.archive]

        # Send to each worker with its own copy of data (to prevent buffer issues)
        for i_work in range(1, n_slave + 1):
            # Each worker gets its own copy to ensure buffer validity
            import copy

            msg = ("archive_update", copy.deepcopy(archive_snapshot))
            self._pending_broadcast_data.append(msg)  # Keep reference alive

            req = comm.isend(msg, dest=i_work, tag=100)
            self._pending_broadcast_requests.append(req)  # Keep request alive

            if verbose:
                print(f"[Master] Queued archive send to worker {i_work}", flush=True)

        if verbose:
            print(f"[Master] Archive broadcast queued (non-blocking)", flush=True)

    def save(self, file_name: str):
        """Save opponent archive to file."""
        archive_path = f"log/{file_name}_archive.pkl"
        with open(archive_path, "wb") as f:
            pickle.dump(self.archive, f)
        print(f"Saved archive with {len(self.archive)} opponents to {archive_path}")

    def load(self, file_name: str):
        """Load opponent archive from file."""
        archive_path = f"log/{file_name}_archive.pkl"
        if os.path.exists(archive_path):
            with open(archive_path, "rb") as f:
                self.archive = pickle.load(f)
            print(
                f"Loaded archive with {len(self.archive)} opponents from {archive_path}"
            )
