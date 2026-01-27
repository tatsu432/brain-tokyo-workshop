#!/usr/bin/env python
"""
Visualize the final trained NEAT network and save as image file.
Simplified for SlimeVolley only - no domain imports to avoid circular dependencies.

Usage:
    python vis/visualize_network.py -i log/test_best.out -o network_visualization.png
"""

import argparse
import os
import sys
from collections import namedtuple

import numpy as np
from matplotlib import pyplot as plt

# Add parent directory to path so we can import neat_src
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# SlimeVolley configuration (hardcoded to avoid domain imports)
Game = namedtuple(
    "Game",
    [
        "env_name",
        "time_factor",
        "actionSelect",
        "input_size",
        "output_size",
        "layers",
        "i_act",
        "h_act",
        "o_act",
        "weightCap",
        "noise_bias",
        "output_noise",
        "max_episode_length",
        "in_out_labels",
    ],
)

slimevolley_config = Game(
    env_name="SlimeVolley-Shaped-v0",
    actionSelect="all",
    input_size=12,
    output_size=3,
    time_factor=0,
    layers=[15, 10],
    i_act=np.full(12, 1),
    h_act=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    o_act=np.full(3, 1),
    weightCap=2.0,
    noise_bias=0.0,
    output_noise=[False] * 3,
    max_episode_length=3000,
    in_out_labels=[
        "agent_x",
        "agent_y",
        "agent_vx",
        "agent_vy",
        "ball_x",
        "ball_y",
        "ball_vx",
        "ball_vy",
        "opponent_x",
        "opponent_y",
        "opponent_vx",
        "opponent_vy",
        "forward",
        "jump",
        "back",
    ],
)

# Create a simple games dict for viewInd compatibility
games = {"slimevolley": slimevolley_config}

# Make games available to viewInd by creating a fake domain.config module
import types

if "domain" not in sys.modules:
    fake_domain = types.ModuleType("domain")
    fake_domain.__path__ = [os.path.join(parent_dir, "domain")]
    sys.modules["domain"] = fake_domain

fake_domain_config = types.ModuleType("domain.config")
fake_domain_config.games = games
sys.modules["domain.config"] = fake_domain_config
sys.modules["domain"].config = fake_domain_config


def main():
    parser = argparse.ArgumentParser(
        description="Visualize NEAT network and save as image"
    )
    parser.add_argument(
        "-i",
        "--infile",
        type=str,
        help="Path to network file (e.g., log/test_best.out)",
        default="log/test_best.out",
    )
    # Removed task argument - only supports SlimeVolley
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output image file path",
        default="network_visualization.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="DPI for output image (higher = better quality, larger file)",
        default=150,
    )

    args = parser.parse_args()

    # Only support SlimeVolley
    task_name = "slimevolley"

    # Load network - import directly from ann.py to avoid circular imports
    import importlib.util
    
    ann_path = os.path.join(parent_dir, "neat_src", "ann.py")
    ann_spec = importlib.util.spec_from_file_location("neat_src_ann", ann_path)
    neat_ann = importlib.util.module_from_spec(ann_spec)
    ann_spec.loader.exec_module(neat_ann)
    importNet = neat_ann.importNet

    try:
        wVec, aVec, wKey = importNet(args.infile)
        # Convert to wMat format expected by viewInd
        # wVec is flattened, we need to reconstruct the matrix
        # For now, we'll use the file directly which viewInd can handle
        print(f"Loaded network from {args.infile}")
        print(f"  Weight vector shape: {wVec.shape}")
        print(f"  Activation vector shape: {aVec.shape}")
        print(f"  Number of connections: {len(wKey)}")
        
        # Diagnostic: Show output node activations
        n_output = slimevolley_config.output_size
        output_activations = aVec[-n_output:].astype(int)
        expected_output_act = slimevolley_config.o_act[0] if len(slimevolley_config.o_act) > 0 else 1
        
        act_names = {
            1: "Linear", 2: "Step", 3: "Sin", 4: "Gauss", 5: "Tanh",
            6: "Sigmoid", 7: "Inverse", 8: "Abs", 9: "ReLU", 10: "Cos", 11: "Squared"
        }
        
        print(f"\n  Output node activations (last {n_output} nodes):")
        for i, act_id in enumerate(output_activations):
            act_name = act_names.get(act_id, f"Unknown({act_id})")
            print(f"    Output {i}: {act_name} (ID: {act_id})")
        
        expected_name = act_names.get(expected_output_act, f"Unknown({expected_output_act})")
        print(f"  Expected (from config): {expected_name} (ID: {expected_output_act})")
        
        if not np.all(output_activations == expected_output_act):
            print(f"  ⚠ Warning: Output activations in saved network don't match config!")
            print(f"    This network was likely trained with different settings.")
    except Exception as e:
        print(f"Error loading network: {e}")
        return 1

    # Create a simple Ind-like object for viewInd
    # viewInd can accept either a file path or an object with wMat and aVec
    class SimpleInd:
        def __init__(self, wVec, aVec):
            # Reconstruct wMat from wVec
            # We need to know the size - try to infer from the file
            ind = np.loadtxt(args.infile, delimiter=",")
            self.wMat = ind[:, :-1]
            self.aVec = aVec

    ind = SimpleInd(wVec, aVec)

    # Visualize
    # Import viewInd here after games is set up in sys.modules
    from vis.viewInd import viewInd

    try:
        print(f"Visualizing network for SlimeVolley")
        fig, ax = viewInd(ind, task_name)
        plt.title("NEAT Network - SlimeVolley", fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Create directory if it doesn't exist
        output_dir = os.path.dirname(args.output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save figure
        print(f"Saving visualization to {args.output}")
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"✓ Successfully saved network visualization to {args.output}")
        plt.close(fig)
        return 0
    except Exception as e:
        print(f"Error visualizing network: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
