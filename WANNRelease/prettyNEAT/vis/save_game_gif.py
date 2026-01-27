#!/usr/bin/env python
"""
Save SlimeVolley game as GIF file where trained NEAT agent plays against baseline.
Simplified for SlimeVolley only - no domain imports to avoid circular dependencies.

Usage:
    python vis/save_game_gif.py -i log/test_best.out -o game_animation.gif --fps 30
"""

import argparse
import os
import sys

import numpy as np
from PIL import Image

# Add parent directory to path so we can import neat_src
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import neat_src functions directly to avoid circular imports
import importlib.util

# Load importNet from ann.py
ann_path = os.path.join(parent_dir, "neat_src", "ann.py")
ann_spec = importlib.util.spec_from_file_location("neat_src_ann", ann_path)
neat_ann = importlib.util.module_from_spec(ann_spec)
ann_spec.loader.exec_module(neat_ann)
importNet = neat_ann.importNet

# Load act and selectAct from ann.py (already loaded above)
# They're in the same module as importNet, so we can get them from neat_ann
act = neat_ann.act
selectAct = neat_ann.selectAct

# SlimeVolley configuration (hardcoded)
SLIMEVOLLEY_INPUT_SIZE = 12
SLIMEVOLLEY_OUTPUT_SIZE = 3
SLIMEVOLLEY_ACTION_SELECT = "all"  # Return all outputs for threshold-based processing


def process_slimevolley_action(action, clip_actions=True, threshold=0.0):
    """
    Convert NEAT continuous action to SlimeVolley binary action.
    Simplified version without domain imports.
    
    Args:
        action: Continuous action from NEAT (3 values: [forward, jump, back])
        clip_actions: Whether to clip actions to [-1, 1]
        threshold: Threshold for activating actions (default: 0.0)
    
    Returns:
        binary_action: numpy array of shape (3,) with binary values [forward, backward, jump]
    """
    action = np.array(action).flatten()
    
    if len(action) == 3:
        if clip_actions:
            action = np.clip(action, -1.0, 1.0)
        # NEAT outputs: [forward, jump, back]
        # SlimeVolley expects: [forward, backward, jump]
        forward = 1 if action[0] > threshold else 0
        jump = 1 if action[1] > threshold else 0
        backward = 1 if action[2] > threshold else 0
        return np.array([forward, backward, jump], dtype=np.int8)
    else:
        raise ValueError(f"Expected 3 outputs, got {len(action)}")


def capture_frame(env):
    """Capture current frame from environment as PIL Image."""
    try:
        # Try to get RGB array from render with rgb_array mode
        frame = env.render(mode="rgb_array")
        if frame is not None and isinstance(frame, np.ndarray):
            # Ensure it's uint8 and has correct shape
            if frame.dtype != np.uint8:
                frame = (
                    (frame * 255).astype(np.uint8)
                    if frame.max() <= 1.0
                    else frame.astype(np.uint8)
                )
            # Handle different array shapes
            if len(frame.shape) == 3:
                return Image.fromarray(frame)
    except Exception:
        pass

    # Fallback: try to access pygame surface directly
    try:
        # Check if environment has a viewer with pygame window
        if hasattr(env, "viewer") and env.viewer is not None:
            if hasattr(env.viewer, "window") and env.viewer.window is not None:
                import pygame

                # Get surface from pygame window
                surface = env.viewer.window
                # Convert pygame surface to numpy array
                frame_string = pygame.image.tostring(surface, "RGB")
                frame = Image.frombytes("RGB", surface.get_size(), frame_string)
                return frame
    except Exception:
        pass

    # Another fallback: try direct pygame screen access
    try:
        if hasattr(env, "screen") and env.screen is not None:
            import pygame

            frame_string = pygame.image.tostring(env.screen, "RGB")
            frame = Image.frombytes("RGB", env.screen.get_size(), frame_string)
            return frame
    except Exception:
        pass

    return None


def save_game_gif(
    infile,
    output_path,
    n_episodes=1,
    fps=30,
    max_steps=3000,
    frame_skip=1,
):
    """
    Run SlimeVolley game with trained NEAT agent and save as GIF.
    Simplified for SlimeVolley only - no domain imports.

    Args:
        infile: Path to trained network file
        output_path: Output GIF file path
        n_episodes: Number of episodes to record (default: 1)
        fps: Frames per second for GIF (default: 30)
        max_steps: Maximum steps per episode (default: 3000)
        frame_skip: Capture every Nth frame (1 = all frames, 2 = every other frame, etc.) (default: 1)
    """
    # Load network
    print(f"Loading network from {infile}")
    wVec, aVec, wKey = importNet(infile)
    wVec = np.copy(wVec)
    wVec[np.isnan(wVec)] = 0

    # Create SlimeVolley environment directly
    try:
        from slimevolleygym.slimevolley import SlimeVolleyEnv
        test_env = SlimeVolleyEnv()
        print("Using SlimeVolleyEnv with baseline opponent")
    except ImportError as e:
        print(f"Error: slimevolleygym not installed. Install with: pip install slimevolleygym")
        print(f"Import error: {e}")
        return False

    # Collect frames
    all_frames = []

    for episode in range(n_episodes):
        print(f"Recording episode {episode + 1}/{n_episodes}...")
        state = test_env.reset()
        episode_frames = []

        done = False
        step = 0
        frames_captured = 0

        for tStep in range(max_steps):
            # Get action from trained network
            annOut = act(wVec, aVec, SLIMEVOLLEY_INPUT_SIZE, SLIMEVOLLEY_OUTPUT_SIZE, state)
            action = selectAct(annOut, SLIMEVOLLEY_ACTION_SELECT)

            # Process action for SlimeVolley
            binary_action = process_slimevolley_action(action, clip_actions=True)

            # Step environment
            state, reward, done, info = test_env.step(binary_action)

            # Only capture frame if we're at the right step (frame skipping)
            if step % frame_skip == 0:
                # Use rgb_array mode directly for faster capture (no display rendering)
                try:
                    rgb_frame = test_env.render(mode="rgb_array")
                    if rgb_frame is not None and isinstance(rgb_frame, np.ndarray):
                        # Ensure it's uint8 and has correct shape
                        if rgb_frame.dtype != np.uint8:
                            rgb_frame = (
                                (rgb_frame * 255).astype(np.uint8)
                                if rgb_frame.max() <= 1.0
                                else rgb_frame.astype(np.uint8)
                            )
                        # Handle different array shapes
                        if len(rgb_frame.shape) == 3:
                            episode_frames.append(Image.fromarray(rgb_frame))
                            frames_captured += 1
                except:
                    # Fallback to capture_frame if rgb_array doesn't work
                    frame = capture_frame(test_env)
                    if frame is not None:
                        episode_frames.append(frame)
                        frames_captured += 1

            step += 1

            # Progress indicator every 500 steps
            if step % 500 == 0:
                print(
                    f"  Step {step}/{max_steps}, captured {frames_captured} frames...",
                    end="\r",
                )

            if done:
                break

        print(f"  Captured {len(episode_frames)} frames from {step} steps")
        all_frames.extend(episode_frames)

    test_env.close()

    # Save as GIF
    if len(all_frames) == 0:
        print("Error: No frames captured. Make sure rendering is working.")
        return False

    print(f"Saving {len(all_frames)} frames as GIF to {output_path}...")

    # Calculate duration per frame (in milliseconds)
    duration_ms = int(1000 / fps)

    # Save GIF
    try:
        all_frames[0].save(
            output_path,
            save_all=True,
            append_images=all_frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"✓ Successfully saved GIF to {output_path}")
        print(f"  Total frames: {len(all_frames)}")
        print(f"  FPS: {fps}")
        print(f"  Duration: {len(all_frames) / fps:.2f} seconds")
        if frame_skip > 1:
            print(f"  Frame skip: {frame_skip} (captured every {frame_skip} steps)")
        return True
    except Exception as e:
        print(f"Error saving GIF: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Save SlimeVolley game as GIF")
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        help="Path to trained network file (e.g., log/test_best.out)",
        default="log/test_best.out",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output GIF file path",
        default="game_animation.gif",
    )
    # Removed task argument - only supports SlimeVolley
    parser.add_argument(
        "--episodes",
        type=int,
        help="Number of episodes to record",
        default=1,
    )
    parser.add_argument(
        "--fps",
        type=int,
        help="Frames per second for GIF",
        default=30,
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Maximum steps per episode",
        default=3000,
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        help="Capture every Nth frame (1=all frames, 2=every other, 3=every third, etc.). Higher values = faster recording but fewer frames.",
        default=1,
    )

    args = parser.parse_args()

    success = save_game_gif(
        args.infile,
        args.output,
        args.episodes,
        args.fps,
        args.max_steps,
        args.frame_skip,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
