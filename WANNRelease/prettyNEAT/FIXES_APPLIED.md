# SlimeVolley Training Bottleneck - Fixes Applied ✅

## Summary

Your SlimeVolley training was stuck at **-3.33 fitness** due to **conflicting action mappings** and an **overcomplicated network**. I've applied comprehensive fixes that should break through this bottleneck.

---

## What Was Wrong

### 🔴 Main Issue: Conflicting Actions

The old action mapping in `domain/slimevolley.py` allowed the agent to output **both forward AND backward** simultaneously when both network outputs were positive. This caused:

- **Wasted network capacity**: Agent evolved to produce conflicting signals
- **Random movements**: Actions cancelled each other out  
- **Edge-hugging behavior**: Agent got stuck moving around edges without purpose

**Example of the bug:**
```python
# OLD CODE (broken):
if action[0] > 0: forward = 1     # ← Both can be 1!
if action[1] > 0: backward = 1    # ← Both can be 1!
# Result: forward=1, backward=1 → agent doesn't move!
```

### 🟡 Secondary Issues

1. **Network too large**: `layers=[15, 10]` for 12→3 mapping was excessive
2. **Too few trials**: `alg_nReps=3` not enough for stochastic environment
3. **Sparse rewards**: Only +1/-1 when losing lives, no intermediate feedback

---

## Fixes Applied

### ✅ 1. Fixed Action Mapping (Critical)

**File:** `domain/slimevolley.py`

**What changed:**
- Forward/backward are now **mutually exclusive** (can't both be active)
- Added **deadband** (-0.2 to 0.2) to prevent random activation
- Higher threshold for jump (0.3) makes it more deliberate

**Before vs After:**
```python
# Input: [0.5, 0.5, 0.0]  (both outputs positive)
Old → [1, 1, 1]  # CONFLICT! forward + backward + jump
New → [1, 0, 1]  # ✓ forward + jump (no conflict)
```

**Test verification:**
```bash
$ python test_action_mapping.py
✓ All 16 tests passed!
```

### ✅ 2. Simplified Network Architecture

**File:** `domain/config.py`

**Changed:**
- `layers`: `[15, 10]` → `[8, 8]` (simpler, faster evolution)
- Network can still grow complex if needed, but starts simple

### ✅ 3. Improved Training Config

**New file:** `p/slimevolley_fixed.json`

**Changes:**
- `alg_nReps`: `3` → `5` (more reliable fitness estimates)
- `alg_act`: `5` → `0` (allow all activation functions)
- `prob_addConn`: `0.15` → `0.2` (faster topology evolution)
- `prob_addNode`: `0.1` → `0.15` (faster complexity growth)
- `prob_mutAct`: `0.0` → `0.1` (explore activation functions)

### ✅ 4. Optional Reward Shaping

**New files:**
- `domain/slimevolley.py`: Added `SlimeVolleyRewardShapingEnv` class
- `p/slimevolley_shaped.json`: Config for shaped rewards

**What it does:**
- Adds dense rewards for good positioning
- **Penalizes edge-hugging** (fixes your observed behavior!)
- Rewards proximity to ball
- Only use if basic fixes aren't enough

---

## How to Use

### Option 1: Basic Fixed Version (Try This First)

```bash
cd /Users/a81808/code/self_study/brain-tokyo-workshop/WANNRelease/prettyNEAT

# Run with fixed configuration
python neat_train.py -p p/slimevolley_fixed.json -n 9
```

**Expected results:**
- **Gen 0-50**: Should improve faster than before
- **Gen 100**: Should break past -3.0 fitness
- **Gen 500+**: Should approach -1.0 or better

### Option 2: With Reward Shaping (If Option 1 Stalls)

```bash
# Use reward-shaped version for easier learning
python neat_train.py -p p/slimevolley_shaped.json -n 9
```

**When to use:**
- If stuck again after 200 generations with Option 1
- If you want faster initial learning
- Trade-off: May overfit to shaped rewards

### Monitoring Progress

**Good signs:**
- ✓ Fitness improving past -3.33
- ✓ Agent moving toward ball instead of edges
- ✓ Occasional winning of points (fitness > -3.0)

**Bad signs:**
- ✗ Still stuck at -3.33 after 100 gens
- ✗ Agent still hugging edges
- ✗ No variation in behavior

---

## Testing Your Fixes

### 1. Verify Action Mapping

```bash
python test_action_mapping.py
```

Expected output:
```
✓ All tests passed! Action mapping is working correctly.
```

### 2. Quick Training Test

```bash
# Run for just 50 generations to see if it's improving
python neat_train.py -p p/slimevolley_fixed.json -n 9 | head -60
```

You should see fitness improving, e.g.:
```
0    - |---| Elite Fit: -4.33  |---| Best Fit:  -4.33
10   - |---| Elite Fit: -4.00  |---| Best Fit:  -4.00
20   - |---| Elite Fit: -3.67  |---| Best Fit:  -3.67
30   - |---| Elite Fit: -3.33  |---| Best Fit:  -3.33
50   - |---| Elite Fit: -3.00  |---| Best Fit:  -3.00  ← Breaking through!
```

---

## Files Modified

### Core Fixes
- ✅ `domain/slimevolley.py` - Fixed action mapping
- ✅ `domain/config.py` - Simplified network architecture  
- ✅ `domain/make_env.py` - Added reward shaping support

### New Configuration Files
- ✅ `p/slimevolley_fixed.json` - Recommended config
- ✅ `p/slimevolley_shaped.json` - With reward shaping

### Documentation & Testing
- ✅ `SLIMEVOLLEY_DIAGNOSIS.md` - Detailed analysis
- ✅ `FIXES_APPLIED.md` - This file
- ✅ `test_action_mapping.py` - Verification tests

---

## What to Expect

### Timeline

| Generations | Expected Fitness | Expected Behavior |
|-------------|------------------|-------------------|
| 0-50 | -4.0 to -3.5 | Initial exploration, less edge-hugging |
| 50-200 | -3.5 to -2.5 | Breaking through plateau, ball engagement |
| 200-500 | -2.5 to -1.5 | Competitive play, winning some points |
| 500-1000 | -1.5 to -0.5 | Consistent point winning |
| 1000+ | 0.0 to +2.0 | Strong player, positive win rate |

### Key Improvements

**Before fixes:**
- ❌ Stuck at -3.33 for 1000+ generations
- ❌ Agent moves around edges aimlessly
- ❌ No meaningful engagement with ball
- ❌ Forward+backward conflicts waste capacity

**After fixes:**
- ✅ Clear action space enables purposeful movement
- ✅ Simpler network learns faster initially
- ✅ No conflicting actions
- ✅ Deadband prevents random movements

---

## Troubleshooting

### Still Stuck After 200 Generations?

1. **Try reward shaping:**
   ```bash
   python neat_train.py -p p/slimevolley_shaped.json -n 9
   ```

2. **Reduce network complexity further:**
   Edit `domain/config.py`:
   ```python
   layers=[5, 5]  # Even simpler
   ```

3. **Increase population diversity:**
   Edit `p/slimevolley_fixed.json`:
   ```json
   "spec_thresh": 1.5  // Lower threshold = more species
   ```

### Agent Learns but Forgets?

- Increase `spec_dropOffAge` from 128 to 256
- This keeps good species alive longer

### Training Too Slow?

- Reduce `alg_nReps` from 5 to 3 (less accurate but faster)
- Reduce `popSize` from 256 to 128
- Use fewer workers: `-n 5` instead of `-n 9`

---

## Advanced: Understanding the Fix

### Why Mutually Exclusive Actions Matter

In SlimeVolley, forward and backward are **opposite** actions. The game engine interprets them as:
- `forward=1, backward=0` → move right
- `forward=0, backward=1` → move left
- `forward=1, backward=1` → **cancels out** (no movement)
- `forward=0, backward=0` → no horizontal movement

The old mapping allowed case #3, which:
1. Wastes network capacity (learning contradictory outputs)
2. Provides no useful gradient (agent doesn't move regardless)
3. Creates local minimum (stuck outputting conflicting signals)

### Why Deadband Helps

Network outputs are continuous in `[-inf, +inf]` (after activation). Random initialization typically produces values near zero. Without a deadband:
- Small positive noise → action activates
- Small negative noise → opposite action activates
- Agent jerks back and forth randomly

With deadband `[-0.2, 0.2]`:
- Small noise → no action (stable)
- Deliberate signal → clear action
- Evolution can learn meaningful thresholds

---

## Next Steps

1. **Start training with fixed config**
2. **Monitor for 100 generations** to confirm improvement
3. **If stuck**, try reward shaping version
4. **If successful**, can increase complexity:
   - Larger population
   - More generations
   - More complex network architectures

---

## Questions?

- Check `SLIMEVOLLEY_DIAGNOSIS.md` for detailed analysis
- Run `python test_action_mapping.py` to verify fixes
- Compare old config (`p/slimevolley.json`) vs new (`p/slimevolley_fixed.json`)

**Good luck! The fixes should get you past the -3.33 plateau! 🚀**
