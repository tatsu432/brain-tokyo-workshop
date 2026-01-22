"""
Non-Blocking Rendering for Self-Play Training Visualization

This module provides utilities to render SlimeVolley games during training
without blocking the main training process. It uses multiprocessing to
run rendering in a separate process.

Usage in training:
    from domain.render_selfplay import RenderManager
    
    render_manager = RenderManager(render_interval=10)
    
    # In training loop:
    if render_manager.should_render(generation):
        render_manager.render_best(best_wVec, best_aVec, nInput, nOutput, actSelect)

Features:
    - Non-blocking: Training continues while rendering happens
    - Configurable interval: Render every N generations
    - Self-play support: Shows agent vs baseline or archived opponent
    - Process management: Won't spawn multiple renders simultaneously
"""

import os
import sys
import time
import json
import tempfile
import traceback
import subprocess
import logging
import warnings
import numpy as np
from typing import Optional, List, Tuple

# Suppress Gym/NumPy 2.0 compatibility warnings
# These warnings appear when slimevolleygym imports the old gym package
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", message=".*does not support NumPy 2.0.*")
warnings.filterwarnings("ignore", message=".*upgrade to Gymnasium.*")
warnings.filterwarnings("ignore", message=".*contact the authors.*")
warnings.filterwarnings("ignore", message=".*migration guide.*")


class FilteredStderr:
    """
    Custom stderr wrapper that filters out Gym deprecation warnings.
    
    The old gym package prints warnings directly to stderr, bypassing Python's
    warnings system. This class intercepts those writes and filters them out.
    """
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        # Copy attributes from original stderr
        for attr in ['mode', 'encoding', 'errors', 'newlines', 'line_buffering', 'write_through']:
            if hasattr(original_stderr, attr):
                setattr(self, attr, getattr(original_stderr, attr))
    
    def write(self, text):
        # Filter out gym deprecation warnings
        if "Gym has been unmaintained" in text:
            return len(text)  # Pretend we wrote it
        if "does not support NumPy 2.0" in text:
            return len(text)
        if "upgrade to Gymnasium" in text:
            return len(text)
        if "migration guide" in text.lower():
            return len(text)
        if "contact the authors" in text.lower():
            return len(text)
        # Write everything else to original stderr
        return self.original_stderr.write(text)
    
    def flush(self):
        return self.original_stderr.flush()
    
    def close(self):
        return self.original_stderr.close()
    
    def __getattr__(self, name):
        # Delegate all other attributes to original stderr
        return getattr(self.original_stderr, name)


# Set up logger for render operations (defaults to INFO level, DEBUG messages hidden)
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(name)s] %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)  # Default to INFO, so DEBUG messages are hidden


def _save_render_data(
    filepath: str,
    wVec: np.ndarray,
    aVec: np.ndarray,
    nInput: int,
    nOutput: int,
    actSelect: str,
    opponent_data: Optional[Tuple[np.ndarray, np.ndarray]],
    generation: int,
    max_steps: int,
    render_fps: int,
):
    """Save render data to a file for the subprocess to load."""
    data = {
        "wVec": wVec.tolist(),
        "aVec": aVec.tolist(),
        "nInput": nInput,
        "nOutput": nOutput,
        "actSelect": actSelect,
        "generation": generation,
        "max_steps": max_steps,
        "render_fps": render_fps,
        "opponent_wVec": opponent_data[0].tolist() if opponent_data else None,
        "opponent_aVec": opponent_data[1].tolist() if opponent_data else None,
    }
    with open(filepath, "w") as f:
        json.dump(data, f)


def _load_render_data(filepath: str) -> dict:
    """Load render data from file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    # Convert lists back to numpy arrays
    data["wVec"] = np.array(data["wVec"])
    data["aVec"] = np.array(data["aVec"])
    if data["opponent_wVec"] is not None:
        data["opponent_wVec"] = np.array(data["opponent_wVec"])
        data["opponent_aVec"] = np.array(data["opponent_aVec"])

    return data


def _render_episode(
    wVec: np.ndarray,
    aVec: np.ndarray,
    nInput: int,
    nOutput: int,
    actSelect: str,
    opponent_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    generation: int = 0,
    max_steps: int = 3000,
    render_fps: int = 50,
    title_prefix: str = "Gen",
):
    """
    Render a single episode in the current process.

    This function is called by the child process to run the visualization.

    Args:
        wVec: Weight vector of the agent
        aVec: Activation vector of the agent
        nInput: Number of input nodes
        nOutput: Number of output nodes
        actSelect: Action selection method ('prob', 'hard', 'all', etc.)
        opponent_data: Optional (wVec, aVec) tuple for self-play opponent
        generation: Current generation number (for display)
        max_steps: Maximum steps per episode
        render_fps: Target frames per second
        title_prefix: Prefix for window title
    """
    # Suppress pygame welcome message
    os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

    # Force stdout to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
    # Ensure stderr filter is active and unbuffered
    # The filter intercepts Gym warnings that are printed directly to stderr
    # Get the original stderr (unwrap filter if present)
    if isinstance(sys.stderr, FilteredStderr):
        original_stderr = sys.stderr.original_stderr
    else:
        original_stderr = sys.stderr
    # Make stderr unbuffered and wrap with filter
    unbuffered_stderr = os.fdopen(original_stderr.fileno(), "w", buffering=1)
    sys.stderr = FilteredStderr(unbuffered_stderr)

    logger.debug(f"Starting render for generation {generation}")

    # Import slimevolleygym directly to ensure pygame is initialized
    # Suppress warnings during import to avoid Gym/NumPy 2.0 compatibility messages
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import slimevolleygym
            from slimevolleygym.slimevolley import SlimeVolleyEnv

        logger.debug("SlimeVolley imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import slimevolleygym: {e}")
        raise

    # Import NEAT modules
    try:
        from neat_src import act, selectAct

        logger.debug("NEAT modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import neat_src: {e}")
        raise

    # Create environment - use the base SlimeVolleyEnv directly for rendering
    # This avoids any wrapper issues
    env = SlimeVolleyEnv()
    logger.debug("Environment created")

    # Clean weights
    wVec = np.copy(wVec)
    wVec[np.isnan(wVec)] = 0
    wVec[np.isinf(wVec)] = 0

    aVec = np.copy(aVec)
    aVec[np.isnan(aVec)] = 0

    # Prepare opponent if self-play
    opponent_wVec = None
    opponent_aVec = None
    if opponent_data is not None:
        opponent_wVec, opponent_aVec = opponent_data
        opponent_wVec = np.copy(opponent_wVec)
        opponent_wVec[np.isnan(opponent_wVec)] = 0
        opponent_wVec[np.isinf(opponent_wVec)] = 0
        opponent_aVec = np.copy(opponent_aVec)
        opponent_aVec[np.isnan(opponent_aVec)] = 0
    # Run episode
    state = env.reset()

    total_reward = 0
    frame_delay = 1.0 / render_fps
    step = 0

    # Action mapping for discrete actions
    DISCRETE_ACTION_MAP = [
        [0, 1, 0],  # 0: left + no jump
        [0, 0, 0],  # 1: stay + no jump
        [1, 0, 0],  # 2: right + no jump
        [0, 1, 1],  # 3: left + jump
        [0, 0, 1],  # 4: stay + jump
        [1, 0, 1],  # 5: right + jump
    ]

    def process_action(action):
        """Convert network output to binary action."""
        if isinstance(action, (int, np.integer)):
            action_idx = int(action) % 6
            return DISCRETE_ACTION_MAP[action_idx]
        action = np.array(action).flatten()
        if len(action) == 6:
            action_idx = np.argmax(action)
            return DISCRETE_ACTION_MAP[action_idx]
        # Default
        return [0, 0, 0]

    logger.debug("Starting episode loop...")

    try:
        for step in range(max_steps):
            start_time = time.time()

            # Get agent action
            annOut = act(wVec, aVec, nInput, nOutput, state)
            action = selectAct(annOut, actSelect)
            binary_action = process_action(action)

            # Get opponent action (if self-play)
            opp_binary_action = None
            if opponent_wVec is not None:
                # Mirror observation for opponent
                opp_state = _mirror_observation(state)
                opp_annOut = act(
                    opponent_wVec, opponent_aVec, nInput, nOutput, opp_state
                )
                opp_action = selectAct(opp_annOut, actSelect)
                opp_binary_action = process_action(opp_action)

            # Step environment
            if opp_binary_action is not None:
                state, reward, done, info = env.step(binary_action, opp_binary_action)
            else:
                state, reward, done, info = env.step(binary_action)

            total_reward += reward

            # Render - call without mode parameter (render_mode should be set at init)
            env.render()

            # Control frame rate
            elapsed = time.time() - start_time
            if elapsed < frame_delay:
                time.sleep(frame_delay - elapsed)

            if done:
                logger.debug(f"Episode done at step {step}")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error during episode: {e}", exc_info=True)

    # # Print final stats
    # print(f"\n{'='*50}", flush=True)
    # print(f"RENDERED EPISODE: {title_prefix} {generation}", flush=True)
    # print(
    #     f"Agent vs {'Archived Opponent' if opponent_data else 'Baseline'}", flush=True
    # )
    # print(f"Total Reward: {total_reward:.2f}", flush=True)
    # print(f"Steps: {step + 1}", flush=True)
    # print(f"{'='*50}\n", flush=True)

    # Keep window open briefly so user can see final state
    time.sleep(2.0)

    env.close()
    logger.debug("Environment closed")


def _mirror_observation(state: np.ndarray) -> np.ndarray:
    """
    Mirror observation for opponent agent.

    State format:
    [agent_x, agent_y, agent_vx, agent_vy,
     ball_x, ball_y, ball_vx, ball_vy,
     opp_x, opp_y, opp_vx, opp_vy]
    """
    return np.array(
        [
            state[8],  # opp_x becomes agent_x
            state[9],  # opp_y
            -state[10],  # opp_vx (negated)
            state[11],  # opp_vy
            -state[4],  # ball_x (negated)
            state[5],  # ball_y
            -state[6],  # ball_vx (negated)
            state[7],  # ball_vy
            state[0],  # agent_x becomes opp_x
            state[1],  # agent_y
            -state[2],  # agent_vx (negated)
            state[3],  # agent_vy
        ]
    )


class RenderManager:
    """
    Manages non-blocking rendering for training visualization.

    Uses subprocess.Popen to spawn a completely independent Python process,
    avoiding issues with MPI and multiprocessing.

    Usage:
        render_manager = RenderManager(render_interval=10, render_fps=50)

        # In training loop:
        for gen in range(max_gen):
            # ... training code ...

            if render_manager.should_render(gen):
                render_manager.render_best(
                    best_wVec, best_aVec,
                    nInput=12, nOutput=6,
                    actSelect='prob',
                    generation=gen
                )

    The rendering runs in a separate process, so training continues immediately.
    """

    def __init__(
        self,
        render_interval: int = 10,
        render_fps: int = 50,
        max_steps: int = 3000,
        enabled: bool = True,
    ):
        """
        Initialize the render manager.

        Args:
            render_interval: Render every N generations (0 to disable)
            render_fps: Target frames per second for rendering
            max_steps: Maximum steps per rendered episode
            enabled: Whether rendering is enabled
        """
        self.render_interval = render_interval
        self.render_fps = render_fps
        self.max_steps = max_steps
        self.enabled = enabled

        # Track if a render process is currently running
        self._current_process: Optional[subprocess.Popen] = None
        self._temp_file: Optional[str] = None

        # Archive of opponents for self-play rendering
        self._opponent_archive: List[Tuple[np.ndarray, np.ndarray, float, int]] = []

    def _is_process_alive(self) -> bool:
        """Check if the subprocess is still running."""
        if self._current_process is None:
            return False
        return self._current_process.poll() is None

    def _cleanup_process(self):
        """Clean up finished process and temp files."""
        if self._current_process is not None:
            if self._current_process.poll() is not None:
                # Process finished, clean up
                self._current_process = None
                if self._temp_file and os.path.exists(self._temp_file):
                    try:
                        os.remove(self._temp_file)
                    except:
                        pass
                    self._temp_file = None

    def should_render(self, generation: int) -> bool:
        """
        Check if we should render at this generation.

        Args:
            generation: Current generation number

        Returns:
            True if rendering should happen
        """
        if not self.enabled or self.render_interval <= 0:
            return False

        # Check interval
        if generation % self.render_interval != 0:
            return False

        # Clean up finished processes
        self._cleanup_process()

        # Check if a render is already in progress
        if self._is_process_alive():
            logger.debug(
                f"Skipping render at gen {generation}: previous render still running"
            )
            return False

        return True

    def render_best(
        self,
        wVec: np.ndarray,
        aVec: np.ndarray,
        nInput: int,
        nOutput: int,
        actSelect: str,
        generation: int = 0,
        opponent_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_random_archive_opponent: bool = True,
    ):
        """
        Start a non-blocking render of the best individual.

        Args:
            wVec: Weight vector of best individual
            aVec: Activation vector of best individual
            nInput: Number of input nodes
            nOutput: Number of output nodes
            actSelect: Action selection method
            generation: Current generation (for display)
            opponent_data: Optional specific opponent (wVec, aVec) for self-play
            use_random_archive_opponent: If True and archive available, use random opponent
        """
        # Clean up finished process if any
        self._cleanup_process()

        if self._is_process_alive():
            logger.debug("Previous render still running, skipping")
            return

        # Select opponent
        final_opponent_data = opponent_data
        if (
            final_opponent_data is None
            and use_random_archive_opponent
            and self._opponent_archive
        ):
            import random

            # Pick a random opponent from archive
            opp = random.choice(self._opponent_archive)
            final_opponent_data = (opp[0].copy(), opp[1].copy())  # wVec, aVec

        # Save data to temp file
        fd, self._temp_file = tempfile.mkstemp(suffix=".json", prefix="render_data_")
        os.close(fd)

        _save_render_data(
            self._temp_file,
            wVec.copy(),
            aVec.copy(),
            nInput,
            nOutput,
            actSelect,
            final_opponent_data,
            generation,
            self.max_steps,
            self.render_fps,
        )

        # Get path to this module
        module_path = os.path.abspath(__file__)

        # Create environment without MPI variables
        env = os.environ.copy()
        # Remove MPI-related variables to avoid conflicts
        for key in list(env.keys()):
            if "MPI" in key or "PMI" in key or "OMPI" in key:
                del env[key]
        env.pop("IN_MPI", None)

        # Launch subprocess
        self._current_process = subprocess.Popen(
            [sys.executable, module_path, "--render-file", self._temp_file],
            env=env,
            stdout=None,  # Inherit stdout so we can see output
            stderr=None,  # Inherit stderr
        )

        logger.debug(
            f"Started render process (PID: {self._current_process.pid}) for generation {generation}"
        )

    def update_archive(self, archive: List[Tuple[np.ndarray, np.ndarray, float, int]]):
        """
        Update the opponent archive for self-play visualization.

        Args:
            archive: List of (wVec, aVec, fitness, generation) tuples
        """
        self._opponent_archive = (
            [(w.copy(), a.copy(), f, g) for w, a, f, g in archive] if archive else []
        )

    def is_rendering(self) -> bool:
        """Check if a render is currently in progress."""
        return self._is_process_alive()

    def wait_for_render(self, timeout: Optional[float] = None):
        """
        Wait for current render to finish.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)
        """
        if self._current_process is not None and self._is_process_alive():
            try:
                self._current_process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        self._cleanup_process()

    def stop(self):
        """Stop any running render process."""
        if self._current_process is not None and self._is_process_alive():
            logger.debug("Terminating render process...")
            self._current_process.terminate()
            try:
                self._current_process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                logger.debug("Force killing render process...")
                self._current_process.kill()
                self._current_process.wait(timeout=1.0)
        self._cleanup_process()


def render_individual(
    wVec: np.ndarray,
    aVec: np.ndarray,
    nInput: int = 12,
    nOutput: int = 6,
    actSelect: str = "prob",
    opponent_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    generation: int = 0,
    max_steps: int = 3000,
    render_fps: int = 50,
    blocking: bool = True,
) -> Optional[subprocess.Popen]:
    """
    Convenience function to render a single individual.

    Args:
        wVec: Weight vector
        aVec: Activation vector
        nInput: Number of inputs (default 12 for SlimeVolley)
        nOutput: Number of outputs (default 6 for discrete actions)
        actSelect: Action selection method
        opponent_data: Optional (wVec, aVec) for self-play opponent
        generation: Generation number for display
        max_steps: Maximum episode steps
        render_fps: Frames per second
        blocking: If True, wait for render to finish; if False, return immediately

    Returns:
        If blocking=False, returns the Popen object; otherwise None
    """
    if blocking:
        _render_episode(
            wVec,
            aVec,
            nInput,
            nOutput,
            actSelect,
            opponent_data,
            generation,
            max_steps,
            render_fps,
        )
        return None
    else:
        # Use RenderManager for non-blocking
        manager = RenderManager(
            render_interval=1, max_steps=max_steps, render_fps=render_fps
        )
        manager.render_best(
            wVec, aVec, nInput, nOutput, actSelect, generation, opponent_data, False
        )
        return manager._current_process


def _run_from_file(filepath: str):
    """Run rendering from a data file. Called by subprocess."""
    logger.debug(f"Loading data from {filepath}")

    data = _load_render_data(filepath)

    opponent_data = None
    if data["opponent_wVec"] is not None:
        opponent_data = (data["opponent_wVec"], data["opponent_aVec"])

    logger.debug(
        f"Starting render for generation {data['generation']}"
    )

    _render_episode(
        data["wVec"],
        data["aVec"],
        data["nInput"],
        data["nOutput"],
        data["actSelect"],
        opponent_data,
        data["generation"],
        data["max_steps"],
        data["render_fps"],
    )

    logger.debug("Render complete")


if __name__ == "__main__":
    """
    This module can be run in two modes:
    1. As a subprocess for rendering (--render-file <path>)
    2. As a test script (no arguments)
    """
    import argparse

    # Install stderr filter to suppress Gym warnings (must be done before any imports that use gym)
    _original_stderr = sys.stderr
    sys.stderr = FilteredStderr(_original_stderr)

    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    parser = argparse.ArgumentParser(description="SlimeVolley Render Module")
    parser.add_argument(
        "--render-file", type=str, help="Path to render data file (for subprocess mode)"
    )
    parser.add_argument("--test", action="store_true", help="Run test mode")
    args = parser.parse_args()

    if args.render_file:
        # Subprocess mode - render from file
        try:
            _run_from_file(args.render_file)
        except Exception as e:
            logger.error(f"ERROR: {e}", exc_info=True)
            sys.exit(1)
    else:
        # Test mode
        print("=" * 60)
        print("SlimeVolley Render Module - Test Mode")
        print("=" * 60)

        # Create random network weights
        nInput = 12
        nOutput = 6
        nNodes = nInput + nOutput + 10  # Some hidden nodes

        wVec = np.random.randn(nNodes * nNodes) * 0.5
        aVec = np.ones(nNodes)  # Linear activations

        print(f"\nNetwork: {nInput} inputs, {nOutput} outputs, {nNodes} total nodes")
        print(f"Weight vector shape: {wVec.shape}")
        print(f"Activation vector shape: {aVec.shape}")

        # Test blocking render
        print("\n" + "=" * 60)
        print("TEST 1: Blocking render")
        print("  A window should open showing the SlimeVolley game.")
        print("  The test will continue after the episode ends.")
        print("=" * 60 + "\n")

        try:
            render_individual(
                wVec,
                aVec,
                nInput=nInput,
                nOutput=nOutput,
                actSelect="prob",
                generation=0,
                max_steps=500,  # Short episode for testing
                blocking=True,
            )
            print("\nBlocking render completed successfully!")
        except Exception as e:
            print(f"\nBlocking render failed: {e}")
            traceback.print_exc()

        # Test non-blocking render with manager
        print("\n" + "=" * 60)
        print("TEST 2: Non-blocking render with RenderManager")
        print("  A window should open while this process continues.")
        print("=" * 60 + "\n")

        try:
            manager = RenderManager(render_interval=1, max_steps=500, render_fps=30)

            if manager.should_render(0):
                manager.render_best(wVec, aVec, nInput, nOutput, "prob", generation=0)
                print("Render started in background. Main process continuing...\n")

                # Simulate training continuing
                for i in range(10):
                    print(f"  Main process: iteration {i+1}/10...")
                    time.sleep(0.5)
                    # Check if render process is still alive
                    if not manager.is_rendering():
                        print("  Render finished early!")
                        break

                if manager.is_rendering():
                    print("\nWaiting for render to finish...")
                    manager.wait_for_render(timeout=60)

                manager.stop()

            print("\nNon-blocking render test completed!")
        except Exception as e:
            print(f"\nNon-blocking render failed: {e}")
            traceback.print_exc()

        print("\n" + "=" * 60)
        print("All tests complete!")
        print("=" * 60)
