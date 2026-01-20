# SlimeVolley Integration with prettyNEAT

This guide explains how to use the SlimeVolley environment with the prettyNEAT/WANN framework.

## Overview

SlimeVolley is a simple multi-agent volleyball game where two agents compete to get the ball to land on the opponent's side. This integration allows you to train NEAT agents to play SlimeVolley.

**Environment Details:**
- **Observation Space**: 12-dimensional continuous vector
  - `[agent_x, agent_y, agent_vx, agent_vy, ball_x, ball_y, ball_vx, ball_vy, opponent_x, opponent_y, opponent_vx, opponent_vy]`
- **Action Space**: 3 binary actions `[forward, backward, jump]`
- **Reward**: +1 when opponent loses a life, -1 when you lose a life, 0 otherwise
- **Episode Length**: Ends when either agent loses all 5 lives or 3000 timesteps
- **Opponent**: Built-in baseline policy (120-parameter neural network from 2015)

## Installation

1. **Install SlimeVolleyGym** (if not already installed):
   ```bash
   pip install slimevolleygym
   ```

2. **Verify Installation**:
   ```bash
   cd /Users/a81808/code/self_study/brain-tokyo-workshop/WANNRelease/prettyNEAT
   python test_slimevolley.py
   ```

   This will run several tests to ensure the environment is properly integrated.

## Files Created

### Domain Files
- **`domain/slimevolley.py`**: Environment wrapper that adapts SlimeVolley to work with prettyNEAT
  - Handles conversion from NEAT's continuous outputs to SlimeVolley's binary actions
  - Provides both single-agent and self-play variants
  
### Configuration Files
- **`p/slimevolley.json`**: Full training configuration with 256 population size
- **`p/slimevolley_quick.json`**: Faster configuration with 128 population size for testing

### Integration Files
- **`domain/make_env.py`**: Updated to include SlimeVolley environment creation
- **`domain/config.py`**: Updated with SlimeVolley game configuration

## Training

### Quick Test Training (Recommended for First Run)
```bash
# Use 4 CPU cores, runs faster for testing
python neat_train.py -p p/slimevolley_quick.json -n 4
```

**Configuration details:**
- Population: 128 individuals
- Max generations: 512
- Evaluations per individual: 3
- Hidden layers: [15, 10] nodes

### Full Training
```bash
# Use 8+ CPU cores for production training
python neat_train.py -p p/slimevolley.json -n 8
```

**Configuration details:**
- Population: 256 individuals
- Max generations: 2048
- Evaluations per individual: 3
- Hidden layers: [15, 10] nodes
- Species target: 8

### Understanding the Configuration

Key hyperparameters in `slimevolley.json`:

```json
{
  "task": "slimevolley",           // Task name
  "maxGen": 2048,                   // Maximum generations
  "popSize": 256,                   // Population size
  "alg_nReps": 3,                   // Evaluations per individual
  "prob_addConn": 0.15,             // Probability of adding connection
  "prob_addNode": 0.1,              // Probability of adding node
  "spec_target": 8,                 // Target number of species
  "save_mod": 16                    // Save checkpoint every N generations
}
```

## Testing Trained Agents

After training, test your best agent:

```bash
# View the best agent from generation 512
python neat_test.py -p p/slimevolley_quick.json -g 512 -t 10 --view
```

**Arguments:**
- `-p`: Configuration file path
- `-g`: Generation number to load
- `-t`: Number of test trials
- `--view`: Visualize the agent playing

## Architecture

### Network Structure

The NEAT agent uses:
- **Input layer**: 12 nodes (one for each observation dimension)
- **Hidden layers**: Variable topology evolved by NEAT (suggested: 15-25 nodes)
- **Output layer**: 3 nodes (one for each action)
- **Activations**: Multiple activation functions (linear, sigmoid, tanh, relu, etc.)

### Action Conversion

The wrapper converts NEAT's continuous outputs to binary actions:

**Method 1** (Single output value):
```python
forward  = 1 if value > 0.33 else 0
backward = 1 if value < -0.33 else 0
jump     = 1 if abs(value) > 0.5 else 0
```

**Method 2** (Three output values):
```python
forward  = 1 if output[0] > 0 else 0
backward = 1 if output[1] > 0 else 0
jump     = 1 if output[2] > 0 else 0
```

## Expected Results

### Training Progress

Typical training progression against the baseline opponent:

| Generation | Average Reward | Notes |
|-----------|----------------|-------|
| 0-50      | -4.5 to -3.0   | Random behavior, losing quickly |
| 50-200    | -3.0 to -1.0   | Learning basic movement |
| 200-500   | -1.0 to 0.0    | Competitive with baseline |
| 500+      | 0.0 to 1.5     | Beating baseline consistently |

**Note**: Performance can vary significantly based on:
- Random seed
- Population diversity
- Mutation rates
- Number of parallel evaluations

### Benchmark Scores

From the SlimeVolley leaderboard (vs. baseline policy):

| Method | Average Score | Notes |
|--------|---------------|-------|
| PPO | 1.377 ± 1.133 | State-of-the-art RL |
| CMA-ES | 1.148 ± 1.071 | Evolution strategy |
| NEAT/WANN | TBD | Your results here! |

## Advanced Usage

### Self-Play Training

For more advanced training, you can implement self-play by modifying the environment to play against previous versions of your agent:

```python
from domain.slimevolley import SlimeVolleySelfPlayEnv

# Create environment with previous best agent as opponent
env = SlimeVolleySelfPlayEnv(opponent_policy=previous_best_agent)
```

### Visualization

To visualize a trained agent:

```bash
python neat_test.py -p p/slimevolley.json -g 1024 -t 5 --view
```

The visualization will show:
- Your agent (right side)
- Opponent (left side)
- Ball trajectory
- Score display

## Troubleshooting

### Import Error: slimevolleygym not found
```bash
pip install slimevolleygym
```

### Gymnasium vs Gym compatibility
The code uses `gymnasium` (gym v0.26+). If you have issues:
```bash
pip install gymnasium
```

### MPI Not Found
For parallel training:
```bash
pip install mpi4py
```

### Rendering Issues
If rendering fails in headless mode, the environment will continue without visualization. This is expected behavior on remote servers.

## Tips for Better Performance

1. **Population Size**: Larger populations (256-512) explore more diverse solutions
2. **Evaluation Reps**: More evaluations (3-5) give more accurate fitness estimates
3. **Species Target**: More species (8-16) maintain diversity longer
4. **Mutation Rates**: 
   - Increase `prob_addNode` (0.1-0.2) for more complex topologies
   - Increase `prob_addConn` (0.15-0.25) for more connections
5. **Hidden Layers**: Start with [15, 10], increase if performance plateaus
6. **Checkpoint Frequency**: Save every 16-32 generations to track progress

## References

- **SlimeVolleyGym**: https://github.com/hardmaru/slimevolleygym
- **NEAT Algorithm**: Stanley & Miikkulainen (2002) - Evolving Neural Networks through Augmenting Topologies
- **WANN**: Gaier & Ha (2019) - Weight Agnostic Neural Networks

## Next Steps

1. **Run the test**: `python test_slimevolley.py`
2. **Start quick training**: `python neat_train.py -p p/slimevolley_quick.json -n 4`
3. **Monitor progress**: Check `log/` directory for training logs
4. **Visualize results**: Use `neat_test.py` with `--view` flag
5. **Experiment**: Try different hyperparameters in the JSON config files

Good luck evolving your SlimeVolley champion! 🏐
