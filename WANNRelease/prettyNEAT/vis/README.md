# NEAT Visualization Tools

This directory contains tools for visualizing NEAT training results and networks.

## Overview

The visualization tools help you:
1. **Visualize the final trained network** as an image
2. **Save SlimeVolley gameplays as GIF** animations
3. **Visualize training evolution** (nodes, connections, species, scores)

## Understanding Log Files

### Stats File (`*_stats.out`)

The stats file contains training statistics saved by `neat_train.py`. It has the following columns:

| Column | Name | Description |
|--------|------|-------------|
| 0 | `x_scale` | Cumulative evaluation count (not generation index) |
| 1 | `fit_med` | Median fitness in population |
| 2 | `fit_max` | Elite fitness (best in current generation) |
| 3 | `fit_top` | Best fitness ever (across all generations) |
| 4 | `fit_max_raw` | Elite raw game score (actual game reward, not shaped) |
| 5 | `fit_top_raw` | Best raw game score ever |
| 6 | `node_med` | Median number of nodes in population |
| 7 | `conn_med` | Median number of connections in population |

**Note:** The stats file is created by `neat_src/dataGatherer.py` in the `save()` method.

### Species File (`*_spec.out`)

The species file (optional) contains species assignment data:
- Format: `[species_id, fitness]` per individual
- Data is organized sequentially: all individuals from generation 0, then generation 1, etc.
- Used to visualize species evolution over time

### Best Network File (`*_best.out`)

The best network file contains the weight matrix and activation functions:
- Format: CSV with weight matrix + activation vector
- Last column: activation functions
- Other columns: weight matrix (flattened)

## Tools

### 1. Visualize Network (`visualize_network.py`)

Visualizes the final trained NEAT network and saves it as an image file.

**Usage:**
```bash
python vis/visualize_network.py -i log/test_best.out -t slimevolley -o network.png
```

**Arguments:**
- `-i, --infile`: Path to network file (default: `log/test_best.out`)
- `-t, --task`: Task name (default: `slimevolley`)
- `-o, --output`: Output image file path (default: `network_visualization.png`)
- `--dpi`: DPI for output image (default: 150)

**Example:**
```bash
# Visualize the best network from training
python vis/visualize_network.py -i log/test_best.out -t slimevolley -o final_network.png --dpi 200
```

**Output:**
- A PNG image showing the neural network structure
- Nodes are colored by layer
- Edges show connections with weights
- Input/output nodes are labeled

### 2. Save Game GIF (`save_game_gif.py`)

Runs SlimeVolley game with trained NEAT agent and saves as GIF animation.

**Usage:**
```bash
python vis/save_game_gif.py -i log/test_best.out -o game.gif --fps 30
```

**Arguments:**
- `-i, --infile`: Path to trained network file (default: `log/test_best.out`)
- `-o, --output`: Output GIF file path (default: `game_animation.gif`)
- `-t, --task`: Task name (default: `slimevolley`)
- `--episodes`: Number of episodes to record (default: 1)
- `--fps`: Frames per second for GIF (default: 30)
- `--max-steps`: Maximum steps per episode (default: 3000)

**Example:**
```bash
# Record 3 episodes at 30 FPS
python vis/save_game_gif.py -i log/test_best.out -o gameplay.gif --episodes 3 --fps 30
```

**Output:**
- A GIF animation showing the agent playing SlimeVolley
- The agent plays against the built-in baseline opponent
- Each frame shows the game state

**Note:** This requires the environment to support rendering. If rendering fails, check that:
- Display is available (for headless systems, use Xvfb or similar)
- pygame is properly installed
- The environment supports `render(mode="rgb_array")`

### 3. Visualize Training Evolution (`visualize_training.py`)

Creates comprehensive visualization of NEAT training evolution.

**Usage:**
```bash
python vis/visualize_training.py -s log/test_stats.out -o training.png
```

**Arguments:**
- `-s, --stats`: Path to stats.out file (default: `log/test_stats.out`)
- `--spec`: Optional path to spec.out file for species data
- `-o, --output`: Output image file path (default: `training_evolution.png`)
- `--dpi`: DPI for output image (default: 150)

**Example:**
```bash
# Full visualization with species data
python vis/visualize_training.py -s log/test_stats.out --spec log/test_spec.out -o training.png --dpi 200
```

**Output:**
A multi-panel figure showing:
1. **Fitness Evolution (Shaped Reward)**: Elite, best, and median fitness over generations
2. **Game Score Evolution (Raw Reward)**: Actual game scores (not shaped fitness)
3. **Network Complexity - Nodes**: Median number of nodes over time
4. **Network Complexity - Connections**: Median number of connections over time
5. **Species Evolution**: Number of species over generations (if spec file provided)

## Complete Workflow Example

After training with `neat_train.py`:

```bash
# 1. Visualize the final network
python vis/visualize_network.py -i log/test_best.out -t slimevolley -o results/final_network.png

# 2. Create gameplay GIF
python vis/save_game_gif.py -i log/test_best.out -o results/gameplay.gif --fps 30

# 3. Visualize training evolution
python vis/visualize_training.py -s log/test_stats.out --spec log/test_spec.out -o results/training_evolution.png
```

## Troubleshooting

### Network Visualization Issues

- **Import errors**: Make sure you're running from the `prettyNEAT` directory
- **Task not found**: Check available tasks in `domain/config.py`
- **Network file format**: Ensure the network file is in the correct format (CSV with weights + activations)

### GIF Creation Issues

- **No frames captured**: 
  - Check that rendering is working: try `python neat_test.py -v True`
  - On headless systems, use Xvfb: `xvfb-run -a python vis/save_game_gif.py ...`
- **Rendering errors**: 
  - Ensure pygame is installed: `pip install pygame`
  - Check display availability: `echo $DISPLAY`

### Training Visualization Issues

- **Missing columns**: The stats file should have 8 columns. If not, some visualizations may be incomplete
- **Species data not showing**: 
  - Provide the `--spec` argument with path to `*_spec.out` file
  - Species count is estimated from the spec file; population size is inferred from stats

## File Locations

After running `neat_train.py` with `-o test`, you'll find:

- `log/test_stats.out`: Training statistics
- `log/test_best.out`: Best network (final)
- `log/test_spec.out`: Species data (if NEAT speciation enabled)
- `log/test_best/`: Directory with best network per generation

## Dependencies

- `matplotlib`: For network and training visualizations
- `PIL/Pillow`: For GIF creation
- `numpy`: For data processing
- `networkx`: For network graph visualization (used by `viewInd.py`)
- `pygame`: For game rendering (used by SlimeVolley)

## Notes

- The stats file uses **evaluations** (not generations) for the x-axis. Each row represents one generation.
- **Raw game scores** (columns 4-5) show actual game performance, not shaped fitness.
- **Species evolution** requires the spec file and population size estimation may not be perfect.
- GIF creation requires a display or virtual display (Xvfb) on headless systems.
