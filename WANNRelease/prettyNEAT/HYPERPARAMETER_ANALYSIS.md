# Hyperparameter Analysis for SlimeVolley Training

## Critical Issues Found

### 1. **Reward Scale Too Small** ⚠️ CRITICAL

**Current:** `survival_scale = 0.01`
- Episodes: 600-700 steps
- Survival reward: 600 * 0.01 = **6.0 to 7.0**
- With curriculum_weight=1.0 (survival only): Total reward = **6-7**

**Problem:**
- Evaluation noise: Only 5 trials (`alg_nReps: 5`)
- Variance in slimevolley is high (episode length varies significantly)
- Signal-to-noise ratio is too low: reward difference of 1-2 is smaller than evaluation variance

**Fix:** Increase `survival_scale` to **0.05-0.1**
- 600 steps * 0.05 = 30.0 reward (much better signal)
- Still small enough to avoid reward hacking
- Better signal-to-noise ratio

### 2. **Evaluation Noise Too High** ⚠️ MAJOR

**Current:** `alg_nReps: 5`
- Only 5 trials per individual
- High variance in slimevolley (episode length, opponent behavior)
- Fitness estimates are noisy

**Problem:**
- With reward scale of 6-7, variance of ±2-3 makes it hard to distinguish good from bad
- NEAT can't reliably select better individuals

**Fix:** Increase to `alg_nReps: 10-15`
- More stable fitness estimates
- Better selection pressure
- Trade-off: Slower training (2-3x more evaluations)

### 3. **Network Too Small** ⚠️ MAJOR

**Current:** `layers=[8, 8]` = 16 hidden neurons
- 12 inputs, 6 outputs
- Complex task requiring ball tracking, positioning, timing

**Problem:**
- Too small for task complexity
- May not have capacity to learn good policies
- NEAT can grow networks, but starting too small slows learning

**Fix:** Use `layers=[15, 10]` or `layers=[20, 15]`
- More capacity for complex behaviors
- Still allows NEAT to evolve topology
- Better starting point

### 4. **Selection Pressure Too High Early** ⚠️ MODERATE

**Current:** `select_cullRatio: 0.3` (removes 30% worst)
- `select_eliteRatio: 0.1` (keeps 10% best)
- `select_tournSize: 4`

**Problem:**
- Early in training, removing 30% might eliminate useful diversity
- High selection pressure can cause premature convergence

**Fix:** Consider reducing to `select_cullRatio: 0.2` early, or use adaptive culling

### 5. **Mutation Rates May Be Too Low** ⚠️ MODERATE

**Current:**
- `prob_addConn: 0.05` (5% chance)
- `prob_addNode: 0.05` (5% chance)
- `prob_mutConn: 0.8` (80% chance)

**Problem:**
- Low connection/node addition rates might slow network growth
- For complex task, might need more exploration

**Fix:** Consider increasing to:
- `prob_addConn: 0.08-0.10`
- `prob_addNode: 0.08-0.10`

### 6. **Speciation Threshold** ⚠️ MODERATE

**Current:**
- `spec_target: 8`
- `spec_thresh: 3.0`
- `spec_threshMin: 0.5`

**Problem:**
- Might be preventing enough diversity
- If all individuals are too similar, exploration suffers

**Fix:** Consider:
- `spec_target: 10-12` (more species)
- `spec_thresh: 2.5` (slightly lower threshold)

## Recommended Configuration

```json
{
    "task": "slimevolley_shaped",
    "enable_curriculum": true,
    "touch_threshold": 10,
    "maxGen": 512,
    "popSize": 128,
    "alg_nReps": 10,              // INCREASED from 5
    "alg_speciate": "neat",
    "alg_probMoo": 0,
    "alg_act": 0,
    "prob_addConn": 0.08,          // INCREASED from 0.05
    "prob_addNode": 0.08,          // INCREASED from 0.05
    "prob_crossover": 0.75,
    "prob_enable": 0.05,
    "prob_mutAct": 0.6,
    "prob_mutConn": 0.8,
    "prob_initEnable": 0.9,
    "select_cullRatio": 0.25,      // REDUCED from 0.3
    "select_eliteRatio": 0.1,
    "select_rankWeight": "exp",
    "select_tournSize": 4,
    "spec_compatMod": 0.25,
    "spec_dropOffAge": 54,
    "spec_target": 10,             // INCREASED from 8
    "spec_thresh": 2.5,            // REDUCED from 3.0
    "spec_threshMin": 0.5,
    "spec_geneCoef": 1,
    "spec_weightCoef": 0.5,
    "save_mod": 16,
    "bestReps": 50
}
```

**And update reward scale in `slimevolley_reward_shaping.py`:**

```python
CURRICULUM_CONFIGS = {
    "survival": {
        "survival_scale": 0.05,    # INCREASED from 0.01
        "curriculum_weight": 1.0,
    },
    "mixed": {
        "survival_scale": 0.05,    # INCREASED from 0.01
        "curriculum_weight": 0.5,
    },
    "wins": {
        "survival_scale": 0.05,    # INCREASED from 0.01
        "curriculum_weight": 0.0,
    },
}
```

**And update network size in `config.py`:**

```python
slimevolley = Game(
    env_name="SlimeVolley-Shaped-v0",
    actionSelect="prob",
    input_size=12,
    output_size=6,
    time_factor=0,
    layers=[15, 10],              # INCREASED from [8, 8]
    # ... rest of config
)
```

## Expected Impact

1. **Reward scale 0.01 → 0.05:**
   - 5x stronger signal
   - Better signal-to-noise ratio
   - Should see faster learning

2. **Evaluation reps 5 → 10:**
   - 2x more stable fitness
   - Better selection
   - Slower but more reliable

3. **Network size [8,8] → [15,10]:**
   - More capacity
   - Better starting point
   - Faster initial learning

4. **Other changes:**
   - More diversity (speciation)
   - Better exploration (mutation rates)
   - Less aggressive selection

## Testing Strategy

1. **First:** Just increase reward scale to 0.05 (easiest, biggest impact)
2. **Second:** Increase evaluation reps to 10
3. **Third:** Increase network size
4. **Fourth:** Adjust other hyperparameters if needed

Start with reward scale change - it's the most likely culprit!
