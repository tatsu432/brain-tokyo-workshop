#!/usr/bin/env python
"""
Visualize NEAT training evolution: nodes, connections, species, elite score, best game score.

Usage:
    python vis/visualize_training.py -s log/test_stats.out -o training_evolution.png
    python vis/visualize_training.py -s log/test_stats.out --spec log/test_spec.out -o training_evolution.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Define lload directly to avoid importing from vis.lplot
# (which would trigger vis/__init__.py -> viewInd -> domain.config circular import)
# lload is just np.loadtxt, so we can define it directly
def lload(fileName):
    """Load data from file using numpy loadtxt (simplified version of vis.lplot.lload)"""
    return np.loadtxt(fileName, delimiter=",")


def load_stats(stats_file):
    """Load stats file and return as numpy array."""
    try:
        stats = lload(stats_file)
        print(f"Loaded stats from {stats_file}")
        print(f"  Shape: {stats.shape}")
        print(f"  Columns: {stats.shape[1]}")
        return stats
    except Exception as e:
        print(f"Error loading stats file: {e}")
        return None


def load_species(spec_file, pop_size=None):
    """
    Load species file and return number of species per generation.

    Args:
        spec_file: Path to spec.out file
        pop_size: Population size (to group individuals by generation)

    Returns:
        num_species_per_gen: Array of species counts per generation, or None if cannot determine
    """
    try:
        spec_data = lload(spec_file)
        if spec_data is None or len(spec_data) == 0:
            return None

        # spec_data format: [species_id, fitness] per individual
        # Data is organized sequentially: all individuals from gen 0, then gen 1, etc.
        print(f"Loaded species data from {spec_file}")
        print(f"  Total individuals: {len(spec_data)}")

        if pop_size is None:
            # Try to infer population size from data
            # Look for patterns in species IDs that might indicate generation boundaries
            # This is a heuristic - if species IDs reset or change significantly, it might be a new generation
            # For now, we'll try to estimate from the total data length
            # A better approach would be to have pop_size passed in
            print("  Warning: Population size not provided, cannot group by generation")
            print("  Showing overall species count only")
            return None

        # Group individuals by generation (assuming pop_size individuals per generation)
        n_generations = len(spec_data) // pop_size
        num_species_per_gen = []

        for gen in range(n_generations):
            start_idx = gen * pop_size
            end_idx = start_idx + pop_size
            gen_spec_ids = spec_data[start_idx:end_idx, 0].astype(int)
            unique_species = np.unique(gen_spec_ids)
            num_species_per_gen.append(len(unique_species))

        print(f"  Calculated species count for {n_generations} generations")
        print(f"  Average species per generation: {np.mean(num_species_per_gen):.2f}")
        return np.array(num_species_per_gen)
    except Exception as e:
        print(f"Warning: Could not load species file: {e}")
        return None


def visualize_training(
    stats_file, spec_file=None, output_path="training_evolution.png", dpi=150
):
    """
    Create comprehensive visualization of NEAT training evolution.

    Args:
        stats_file: Path to stats.out file
        spec_file: Optional path to spec.out file for species data
        output_path: Output image file path
        dpi: DPI for output image
    """
    # Load stats
    stats = load_stats(stats_file)
    if stats is None:
        return False

    # Parse stats columns based on dataGatherer format:
    # [x_scale, fit_med, fit_max, fit_top, fit_max_raw, fit_top_raw, node_med, conn_med]
    n_cols = stats.shape[1]

    if n_cols < 8:
        print(f"Warning: Expected at least 8 columns, got {n_cols}")
        print("Some visualizations may be incomplete.")

    # Extract data
    generations = np.arange(len(stats))  # Generation index
    evaluations = stats[:, 0] if n_cols > 0 else generations  # x_scale (evaluations)

    # Fitness metrics
    fit_med = stats[:, 1] if n_cols > 1 else None  # Median fitness
    fit_max = stats[:, 2] if n_cols > 2 else None  # Elite fitness (max in generation)
    fit_top = stats[:, 3] if n_cols > 3 else None  # Best fitness ever

    # Raw game scores (actual game reward, not shaped fitness)
    fit_max_raw = stats[:, 4] if n_cols > 4 else None  # Elite raw game score
    fit_top_raw = stats[:, 5] if n_cols > 5 else None  # Best raw game score ever

    # Complexity metrics
    node_med = stats[:, 6] if n_cols > 6 else None  # Median nodes
    conn_med = stats[:, 7] if n_cols > 7 else None  # Median connections

    # Load species data if available
    num_species_per_gen = None
    if spec_file:
        # Try to infer population size from stats
        # If we have evaluation counts, we can estimate pop size
        # pop_size ≈ (evaluations[1] - evaluations[0]) for first generation
        pop_size = None
        if len(evaluations) > 1:
            # Estimate population size from first generation's evaluation count
            pop_size = int(evaluations[0]) if evaluations[0] > 0 else None
            if pop_size is None and len(evaluations) > 1:
                pop_size = int(evaluations[1] - evaluations[0])

        num_species_per_gen = load_species(spec_file, pop_size)

        # If we got species data but it doesn't match stats length, pad or trim
        if num_species_per_gen is not None and len(num_species_per_gen) > 0:
            if len(num_species_per_gen) < len(stats):
                # Pad with last value
                last_value = num_species_per_gen[-1] if len(num_species_per_gen) > 0 else 1
                num_species_per_gen = np.append(
                    num_species_per_gen,
                    np.full(
                        len(stats) - len(num_species_per_gen), last_value
                    ),
                )
            elif len(num_species_per_gen) > len(stats):
                # Trim to match
                num_species_per_gen = num_species_per_gen[: len(stats)]
        elif num_species_per_gen is not None and len(num_species_per_gen) == 0:
            # Empty array, set to None
            num_species_per_gen = None

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Fitness Evolution (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    if fit_max is not None:
        ax1.plot(generations, fit_max, label="Elite Fitness", linewidth=2, color="blue")
    if fit_top is not None:
        ax1.plot(
            generations,
            fit_top,
            label="Best Fitness",
            linewidth=2,
            color="red",
            linestyle="--",
        )
    if fit_med is not None:
        ax1.plot(
            generations,
            fit_med,
            label="Median Fitness",
            linewidth=1,
            color="gray",
            alpha=0.7,
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Shaped)")
    ax1.set_title("Fitness Evolution (Shaped Reward)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Game Score Evolution (Top Right) - Raw game scores
    ax2 = fig.add_subplot(gs[0, 1])
    if fit_max_raw is not None:
        ax2.plot(
            generations,
            fit_max_raw,
            label="Elite Game Score",
            linewidth=2,
            color="green",
        )
    if fit_top_raw is not None:
        ax2.plot(
            generations,
            fit_top_raw,
            label="Best Game Score",
            linewidth=2,
            color="darkgreen",
            linestyle="--",
        )
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Game Score (Raw)")
    ax2.set_title("Game Score Evolution (Raw Reward)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Network Complexity - Nodes (Bottom Left, Top)
    ax3 = fig.add_subplot(gs[1, 0])
    if node_med is not None:
        ax3.plot(
            generations, node_med, label="Median Nodes", linewidth=2, color="purple"
        )
        ax3.fill_between(
            generations, node_med * 0.9, node_med * 1.1, alpha=0.2, color="purple"
        )
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Number of Nodes")
    ax3.set_title("Network Complexity: Nodes")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Network Complexity - Connections (Bottom Left, Bottom)
    ax4 = fig.add_subplot(gs[1, 1])
    if conn_med is not None:
        ax4.plot(
            generations,
            conn_med,
            label="Median Connections",
            linewidth=2,
            color="orange",
        )
        ax4.fill_between(
            generations, conn_med * 0.9, conn_med * 1.1, alpha=0.2, color="orange"
        )
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Number of Connections")
    ax4.set_title("Network Complexity: Connections")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Species Evolution (Bottom Right)
    ax5 = fig.add_subplot(gs[2, :])
    if num_species_per_gen is not None:
        ax5.plot(
            generations,
            num_species_per_gen,
            label="Number of Species",
            linewidth=2,
            color="brown",
        )
        ax5.set_ylabel("Number of Species")
        ax5.legend()
    else:
        # If we don't have per-generation data, show a message
        ax5.text(
            0.5,
            0.5,
            "Species data not available\n(use --spec to provide species file)",
            ha="center",
            va="center",
            transform=ax5.transAxes,
            fontsize=12,
        )
        ax5.set_ylabel("Number of Species")
    ax5.set_xlabel("Generation")
    ax5.set_title("Species Evolution")
    ax5.grid(True, alpha=0.3)

    # Add overall title
    fig.suptitle("NEAT Training Evolution", fontsize=16, fontweight="bold", y=0.995)

    # Save figure
    print(f"Saving visualization to {output_path}")
    try:
        plt.tight_layout()
    except:
        # If tight_layout fails, just save without it
        pass
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"✓ Successfully saved training visualization to {output_path}")
    plt.close(fig)

    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize NEAT training evolution")
    parser.add_argument(
        "-s",
        "--stats",
        type=str,
        help="Path to stats.out file (e.g., log/test_stats.out)",
        default="log/test_stats.out",
    )
    parser.add_argument(
        "--spec",
        type=str,
        help="Optional path to spec.out file for species data",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output image file path",
        default="training_evolution.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="DPI for output image",
        default=150,
    )

    args = parser.parse_args()

    success = visualize_training(
        args.stats,
        args.spec,
        args.output,
        args.dpi,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
