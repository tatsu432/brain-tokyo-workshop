# NEAT Log Format Explanation

## How Logs Are Created

### During Training (`neat_train.py`)

When you run `neat_train.py`, the training process creates several log files:

1. **Stats File** (`log/<prefix>_stats.out`):
   - Created by `neat_src/dataGatherer.py` in the `save()` method
   - Called at the end of each generation (or periodically)
   - Contains aggregated statistics for each generation

2. **Best Network File** (`log/<prefix>_best.out`):
   - Contains the best individual found so far
   - Updated whenever a new best is found
   - Format: CSV with weight matrix + activation vector

3. **Species File** (`log/<prefix>_spec.out`):
   - Created if NEAT speciation is enabled (`alg_speciate == "neat"`)
   - Contains species assignments for each individual
   - Format: `[species_id, fitness]` per individual

### During Testing (`neat_test.py`)

When you run `neat_test.py`, it creates:

1. **Fitness Distribution** (`log/result_fitDist.out`):
   - Contains fitness values from test runs
   - Used to evaluate network performance

2. **Raw Score** (`log/result_rawScore.out`):
   - For SlimeVolley: contains actual game scores
   - Shows performance without reward shaping

## Stats File Format

The stats file (`*_stats.out`) is a CSV file with 12 columns:

```
Column 0: x_scale          - Cumulative evaluation count
Column 1: fit_med           - Median fitness in population
Column 2: fit_max           - Elite fitness (best in generation)
Column 3: fit_top           - Best fitness ever (across all generations)
Column 4: fit_max_raw       - Elite raw game score
Column 5: fit_top_raw       - Best raw game score ever
Column 6: node_med          - Median number of nodes
Column 7: conn_med          - Median number of connections
Column 8: node_elite        - Number of nodes in elite network (best in generation)
Column 9: conn_elite        - Number of connections in elite network
Column 10: node_best        - Number of nodes in best network (best ever)
Column 11: conn_best        - Number of connections in best network
```

### Understanding the Columns

**x_scale (Column 0):**
- This is NOT the generation number
- It's the cumulative number of evaluations (individuals tested)
- Example: If population size is 100, generation 0 has x_scale=100, generation 1 has x_scale=200, etc.
- Formula: `x_scale[gen] = sum(pop_size for all previous generations)`

**Fitness vs Game Score:**
- `fit_max` / `fit_top` (columns 2-3): **Shaped fitness** - includes reward shaping, curriculum learning, etc.
- `fit_max_raw` / `fit_top_raw` (columns 4-5): **Raw game score** - actual game performance
  - For SlimeVolley: This is the actual score (rallies won - rallies lost)
  - This is what you want to visualize for "game score evolution"

**Complexity Metrics:**
- `node_med` (column 6): Median number of nodes in the population
- `conn_med` (column 7): Median number of connections in the population
- These show how the network complexity evolves over time

## Species File Format

The species file (`*_spec.out`) contains species assignment data:

**Actual Format (Saved by dataGatherer.py):**
```
Format: Transposed matrix (2 rows, total_individuals columns)
- Row 0: All species IDs across ALL generations [species_id_0, ..., species_id_n]
- Row 1: All fitness values across ALL generations [fitness_0, ..., fitness_n]
- Shape: (2, total_individuals)
- Data organization: All individuals from gen 0, then all from gen 1, then gen 2, etc.
```

**Note:** The file is saved in transposed format and accumulates data across all generations. When loading, transpose it to get:
```
Format: [species_id, fitness] per individual
- Each row represents one individual
- Data is organized sequentially: all individuals from gen 0, then gen 1, etc.
- Column 0: species_id - Integer ID of the species this individual belongs to
- Column 1: fitness - Fitness value of this individual
- Shape after transpose: (total_individuals, 2)
```

**Extracting Species Count Per Generation:**
1. Transpose the data to get (total_individuals, 2) format
2. Group individuals by generation using population size:
   - Generation 0: individuals [0 : pop_size]
   - Generation 1: individuals [pop_size : 2*pop_size]
   - Generation 2: individuals [2*pop_size : 3*pop_size]
   - etc.
3. Count unique species IDs in each generation
4. This gives you the number of species over time

**Important:** The spec file now accumulates data across all generations (fixed in dataGatherer.py). Each call to `gatherData()` appends the current generation's data to the accumulated data, so the file contains the full history of species assignments.

## Data Flow

```
neat_train.py
    │
    ├─> EvolutionRunner
    │       │
    │       └─> DataGatherer.gatherData()  [each generation]
    │               │
    │               ├─> Updates: fit_max, fit_top, node_med, conn_med, num_species
    │               └─> Stores: elite, best individuals
    │
    └─> DataGatherer.save()  [periodically or at end]
            │
            ├─> Saves: *_stats.out (8 columns)
            ├─> Saves: *_best.out (best network)
            └─> Saves: *_spec.out (species data, if enabled)
```

## Code Locations

- **Data Collection**: `neat_src/dataGatherer.py`
  - `gatherData()`: Collects data each generation
  - `save()`: Saves data to files

- **Training Loop**: `application/evolution_runner.py`
  - Calls `gatherData()` after each generation
  - Calls `save()` periodically

- **Log Format**: Defined in `dataGatherer.py` line 504-532

## Example: Reading Stats File

```python
import numpy as np
from vis.lplot import lload

# Load stats
stats = lload("log/test_stats.out")

# Extract columns
generations = np.arange(len(stats))
evaluations = stats[:, 0]        # x_scale
fit_med = stats[:, 1]            # Median fitness
fit_max = stats[:, 2]            # Elite fitness
fit_top = stats[:, 3]            # Best fitness
fit_max_raw = stats[:, 4]        # Elite game score
fit_top_raw = stats[:, 5]        # Best game score
node_med = stats[:, 6]           # Median nodes
conn_med = stats[:, 7]           # Median connections

# Plot game score evolution
import matplotlib.pyplot as plt
plt.plot(generations, fit_top_raw, label="Best Game Score")
plt.xlabel("Generation")
plt.ylabel("Game Score")
plt.title("Game Score Evolution")
plt.legend()
plt.show()
```

## Key Insights

1. **Use `fit_top_raw` for game score visualization** - This shows actual game performance
2. **Use `fit_top` for fitness visualization** - This shows shaped fitness (may include curriculum learning)
3. **`x_scale` is cumulative evaluations** - Not generation number, but related
4. **Species count requires spec file** - Not stored in stats file directly
5. **Network complexity increases over time** - Watch `node_med` and `conn_med` grow

## Visualization Tools

Use the provided visualization tools:

1. `visualize_network.py` - Visualize final network structure
2. `save_game_gif.py` - Create gameplay animations
3. `visualize_training.py` - Comprehensive training evolution plots

See `README.md` for usage instructions.
