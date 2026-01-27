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

        # The spec file is saved in transposed format:
        # - Row 0: All species IDs
        # - Row 1: All fitness values
        # We need to transpose it to get [species_id, fitness] per individual
        if spec_data.shape[0] == 2 and spec_data.shape[1] > 2:
            # Transposed format: (2, n_individuals) -> transpose to (n_individuals, 2)
            spec_data = spec_data.T
            print(f"Loaded species data from {spec_file} (transposed format)")
        elif spec_data.shape[1] == 2:
            # Already in correct format: (n_individuals, 2)
            print(f"Loaded species data from {spec_file} (standard format)")
        else:
            print(f"Warning: Unexpected spec file shape: {spec_data.shape}")
            return None

        # spec_data format: [species_id, fitness] per individual
        # Data is organized sequentially: all individuals from gen 0, then gen 1, etc.
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
        
        if n_generations == 0:
            # Spec file has fewer individuals than expected pop_size
            # This might be from a partial run or different configuration
            # Try to use the actual number of individuals as a single "generation"
            print(f"  Warning: Spec file has {len(spec_data)} individuals, but pop_size is {pop_size}")
            print(f"  This suggests the spec file may be incomplete or from a different run")
            print(f"  Attempting to calculate species count from available data...")
            
            # If we have at least some data, try to group it
            if len(spec_data) > 0:
                # Use the actual number of individuals as if it's one generation
                unique_species = np.unique(spec_data[:, 0].astype(int))
                num_species = len(unique_species)
                print(f"  Found {num_species} unique species in {len(spec_data)} individuals")
                # Return a single value that will be repeated for all generations in stats
                return np.array([num_species])
            else:
                return None
        
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
    # [x_scale, fit_med, fit_max, fit_top, fit_max_raw, fit_top_raw, 
    #  node_med, conn_med, node_elite, conn_elite, node_best, conn_best]
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

    # Complexity metrics - median
    node_med = stats[:, 6] if n_cols > 6 else None  # Median nodes
    conn_med = stats[:, 7] if n_cols > 7 else None  # Median connections
    
    # Complexity metrics - elite and best (new columns)
    node_elite = stats[:, 8] if n_cols > 8 else None  # Elite network nodes
    conn_elite = stats[:, 9] if n_cols > 9 else None  # Elite network connections
    node_best = stats[:, 10] if n_cols > 10 else None  # Best network nodes
    conn_best = stats[:, 11] if n_cols > 11 else None  # Best network connections

    # Load species data if available
    num_species_per_gen = None
    if spec_file:
        # Try to infer population size from stats
        # If we have evaluation counts, we can estimate pop size
        # pop_size ≈ (evaluations[1] - evaluations[0]) for first generation
        pop_size = None
        if len(evaluations) > 1:
            # Estimate population size from difference between consecutive evaluations
            pop_size = int(evaluations[1] - evaluations[0])
        elif len(evaluations) == 1 and evaluations[0] > 0:
            # Single generation: pop_size equals the evaluation count
            pop_size = int(evaluations[0])

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
    # Consistent styling: Best (red dotted), Elite (blue solid), Median (gray thin)
    ax1 = fig.add_subplot(gs[0, 0])
    handles1 = []
    if fit_top is not None:
        line1 = ax1.plot(
            generations,
            fit_top,
            label="Best Fitness",
            linewidth=2,
            color="red",
            linestyle="--",
        )
        handles1.append(line1[0])
    if fit_max is not None:
        line2 = ax1.plot(generations, fit_max, label="Elite Fitness", linewidth=2, color="blue")
        handles1.append(line2[0])
    if fit_med is not None:
        line3 = ax1.plot(
            generations,
            fit_med,
            label="Median Fitness",
            linewidth=1,
            color="gray",
            alpha=0.7,
        )
        handles1.append(line3[0])
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Fitness (Shaped)")
    ax1.set_title("Fitness Evolution (Shaped Reward)")
    ax1.legend(handles=handles1)
    ax1.grid(True, alpha=0.3)

    # 2. Game Score Evolution (Top Right) - Raw game scores
    # Consistent styling: Best (red dotted), Elite (blue solid), Median (gray thin)
    ax2 = fig.add_subplot(gs[0, 1])
    handles2 = []
    if fit_top_raw is not None:
        line1 = ax2.plot(
            generations,
            fit_top_raw,
            label="Best Game Score",
            linewidth=2,
            color="red",
            linestyle="--",
        )
        handles2.append(line1[0])
    if fit_max_raw is not None:
        line2 = ax2.plot(
            generations,
            fit_max_raw,
            label="Elite Game Score",
            linewidth=2,
            color="blue",
        )
        handles2.append(line2[0])
    # Note: No median for raw game scores currently tracked
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Game Score (Raw)")
    ax2.set_title("Game Score Evolution (Raw Reward)")
    ax2.legend(handles=handles2)
    ax2.grid(True, alpha=0.3)

    # 3. Network Complexity - Nodes (Bottom Left, Top)
    # Consistent styling: Best (red dotted), Elite (blue solid), Median (gray thin)
    ax3 = fig.add_subplot(gs[1, 0])
    handles3 = []
    if node_best is not None:
        line1 = ax3.plot(
            generations, node_best, label="Best Nodes", linewidth=2, color="red", linestyle="--"
        )
        handles3.append(line1[0])
    if node_elite is not None:
        line2 = ax3.plot(
            generations, node_elite, label="Elite Nodes", linewidth=2, color="blue", linestyle="-"
        )
        handles3.append(line2[0])
    if node_med is not None:
        line3 = ax3.plot(
            generations, node_med, label="Median Nodes", linewidth=1, color="gray", alpha=0.7
        )
        handles3.append(line3[0])
    ax3.set_xlabel("Generation")
    ax3.set_ylabel("Number of Nodes")
    ax3.set_title("Network Complexity: Nodes")
    ax3.legend(handles=handles3)
    ax3.grid(True, alpha=0.3)

    # 4. Network Complexity - Connections (Bottom Left, Bottom)
    # Consistent styling: Best (red dotted), Elite (blue solid), Median (gray thin)
    ax4 = fig.add_subplot(gs[1, 1])
    handles4 = []
    if conn_best is not None:
        line1 = ax4.plot(
            generations, conn_best, label="Best Connections", linewidth=2, color="red", linestyle="--"
        )
        handles4.append(line1[0])
    if conn_elite is not None:
        line2 = ax4.plot(
            generations, conn_elite, label="Elite Connections", linewidth=2, color="blue", linestyle="-"
        )
        handles4.append(line2[0])
    if conn_med is not None:
        line3 = ax4.plot(
            generations,
            conn_med,
            label="Median Connections",
            linewidth=1,
            color="gray",
            alpha=0.7,
        )
        handles4.append(line3[0])
    ax4.set_xlabel("Generation")
    ax4.set_ylabel("Number of Connections")
    ax4.set_title("Network Complexity: Connections")
    ax4.legend(handles=handles4)
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
