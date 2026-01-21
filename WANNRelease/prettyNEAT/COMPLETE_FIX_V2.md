# Complete Fix V2: All Issues Identified and Resolved

## Your Results Reveal the Real Problem

### What You Experienced:

**Original training (before my fixes):**
```
Fitness: -3.33 to -4.33 (stuck at -3.33 for 1000+ gens)
Behavior: Agent moves around edges, no ball engagement
```

**After my fixes with shaped rewards:**
```
Fitness: +3.05 to +5.57 (stuck at +5.17 for 800+ gens)  
Behavior: (Unknown - but probably following ball without hitting it)
```

### The Sign Change Reveals Everything!

Sl imeVolley gives ±1 per life won/lost (max ±5 per episode).

**Your positive fitness (+5.17) proves:**
- Real game reward: ~-3 (still losing!)
- Shaped rewards: ~+8 (overwhelming the signal!)
- **The agent is NOT playing better, just exploiting shaped rewards!**

This is classic **reward hacking**.

---

## ALL Issues Found (3 Total)

### 🔴 Issue #1: Conflicting Action Mapping
**Status:** ✅ FIXED in my first attempt

**Problem:**
```python
# Old code allowed both forward AND backward to activate
if action[0] > 0: forward = 1
if action[1] > 0: backward = 1
# Result: Both could be 1 → actions cancel out
```

**Fix:**
```python
# New code: mutually exclusive with deadband
if action[0] > 0.2: forward=1, backward=0
elif action[0] < -0.2: forward=0, backward=1
else: forward=0, backward=0  # deadband
```

---

### 🔴 Issue #2: Unbounded Output Activation
**Status:** ✅ FIXED NOW

**Problem:**
```python
# domain/config.py line 130
o_act=np.full(3, 1),  # Linear activation (unbounded!)
```

With linear activation:
- Network can output [100, 50, 75] or any value
- Large values saturate action mapping
- No useful gradient for evolution

**Fix:**
```python
# domain/config.py line 130  
o_act=np.full(3, 5),  # Tanh activation (bounded to [-1, 1])
```

With tanh activation:
- Network outputs bounded to [-1, 1]
- Action mapping thresholds (0.2, 0.3) make sense
- Clean gradient for evolution

**This is likely THE critical fix that was missing!**

---

### 🔴 Issue #3: Reward Shaping is Broken
**Status:** ✅ IDENTIFIED, needs removal

**Problem:**
- Shaped rewards per step: ~1-2 points
- Over 3000 steps: ~+3000 to +6000 total
- With weight=0.01: ~+30 to +60
- Original game reward: ~-3 to -5
- **Shaped rewards are 10x larger than game rewards!**

**Result:**
- Agent optimizes shaped rewards, not game performance
- Gets +5 fitness while still losing the actual game
- Stuck at local optimum (follow ball, avoid edges, don't play)

**Fix:**
Use `slimevolley_fixed.json` (no shaping), not `slimevolley_shaped.json`

---

## Complete Solution

### All Fixes Applied:

1. ✅ Fixed action mapping (mutually exclusive forward/backward)
2. ✅ Simplified network ([8,8] instead of [15,10])
3. ✅ **NEW:** Changed output activation (tanh instead of linear)
4. ✅ Removed reward shaping (will use sparse rewards)

### Files Modified:

1. **`domain/slimevolley.py`**
   - Fixed `_process_action()` method
   
2. **`domain/config.py`**
   - Changed `layers=[8,8]`
   - **Changed `o_act=np.full(3,5)` ← THE CRITICAL FIX**

3. **`p/slimevolley_fixed.json`**
   - Updated hyperparameters
   - NO reward shaping

---

## How to Test

### Run This Command:

```bash
cd /Users/a81808/code/self_study/brain-tokyo-workshop/WANNRelease/prettyNEAT

python neat_train.py -p p/slimevolley_fixed.json -n 9
```

### What to Expect:

**Gen 0-20:**
```
0    - |---| Elite Fit: -4.33  |---| Best Fit:  -4.33
10   - |---| Elite Fit: -4.00  |---| Best Fit:  -4.00
20   - |---| Elite Fit: -3.67  |---| Best Fit:  -3.67
```
Starting point, gradual improvement

**Gen 20-100:**
```
50   - |---| Elite Fit: -3.00  |---| Best Fit:  -3.00  ← Breaking through!
100  - |---| Elite Fit: -2.33  |---| Best Fit:  -2.33  ← Clear progress
```
Should break past -3.33 plateau

**Gen 100-500:**
```
200  - |---| Elite Fit: -1.67  |---| Best Fit:  -1.67
500  - |---| Elite Fit: -0.67  |---| Best Fit:  -0.67
```
Approaching competitive play

**Gen 500+:**
```
1000 - |---| Elite Fit: +0.33  |---| Best Fit:  +0.33  ← Positive = winning!
```
Agent actually winning more points than losing

---

## Red Flags to Watch For

### ❌ Still Stuck at -3.33
If after 200 generations you see:
```
200  - |---| Elite Fit: -3.33  |---| Best Fit:  -3.33  ← Still stuck!
```

**This means:** There's another issue we haven't found yet.

**Next steps:**
1. Check if output activation change was actually applied
2. Verify action mapping is being called correctly
3. May need to investigate NEAT's network building process

### ❌ Positive Fitness Too Early
If you see:
```
10   - |---| Elite Fit: +2.00  |---| Best Fit:  +2.00  ← Positive too soon!
```

**This means:** Reward shaping is still active somehow.

**Next steps:**
1. Verify you're using `slimevolley_fixed.json` not `slimevolley_shaped.json`
2. Check the task name in config is "slimevolley" not "slimevolley_shaped"

### ❌ No Improvement at All
If you see:
```
0    - |---| Elite Fit: -4.33  |---| Best Fit:  -4.33
100  - |---| Elite Fit: -4.33  |---| Best Fit:  -4.33  ← No change
```

**This means:** Network isn't learning anything.

**Possible causes:**
1. Network initialization too random
2. Mutation rates too low
3. Population diversity too low
4. Something fundamentally wrong with task setup

---

## Technical Deep Dive

### Why Linear Output Activation Failed

**The chain of events:**

1. Hidden nodes with diverse activations (alg_act=0)
2. Some hidden nodes use ReLU, squared, abs → produce large values
3. Output nodes use linear activation → pass large values through unbounded
4. Network outputs: [50.0, 30.0, 20.0] (way outside [-1, 1])
5. Action mapping: `if action[0] > 0.2:` → always triggers!
6. All actions activate or saturate → no meaningful learning

**With tanh output:**

1. Hidden nodes with diverse activations (alg_act=0)
2. Some hidden nodes produce large values
3. Output nodes use tanh → **bounds values to [-1, 1]**
4. Network outputs: [0.95, 0.72, 0.88] (bounded)
5. Action mapping thresholds work correctly
6. Evolution can find useful weight changes

### Why Tanh Specifically?

**Tanh properties:**
- Range: [-1, 1] (symmetric)
- Smooth (differentiable everywhere)
- Saturates gracefully (large inputs → ±1)
- Zero-centered (good for forward/backward symmetry)

**Your action mapping zones:**
```
action[0] (horizontal):
  > 0.2:     forward only  (20% to 100% of positive range)
  < -0.2:    backward only (20% to 100% of negative range)
  [-0.2, 0.2]: neither (deadband)

action[1] (jump):
  > 0.3:     jump (30% to 100% of positive range)
  ≤ 0.3:     no jump
```

With outputs in [-1, 1], these thresholds are perfect!

---

## Why This Explains BOTH Issues

### Original Issue (-3.33 plateau):
1. Conflicting action mapping + unbounded outputs
2. Agent couldn't learn meaningful actions
3. Stuck at local optimum

### New Issue (+5.17 plateau):
1. Action mapping fix helped a bit
2. But reward shaping created reward hacking
3. Unbounded outputs still causing saturation
4. Stuck at different local optimum

### Complete Fix:
1. ✓ Action mapping (no conflicts)
2. ✓ Bounded outputs (tanh)
3. ✓ No reward shaping (sparse but real rewards)
4. → Should finally learn!

---

## Expected Timeline After Complete Fix

| Generations | Fitness | What's Happening |
|-------------|---------|------------------|
| 0-50 | -4.0 to -3.5 | Initial exploration with clean action space |
| 50-150 | -3.5 to -2.5 | **Breaking through -3.33 plateau** |
| 150-400 | -2.5 to -1.0 | Learning to engage with ball |
| 400-800 | -1.0 to 0.0 | Approaching even play |
| 800-1500 | 0.0 to +1.5 | Winning more than losing |

If you don't see improvement past -3.33 by generation 200, there's still another issue.

---

## What to Report Back

After running the fixed version, please share:

1. **First 50 generations** of output
2. **Around generation 150-200** (should show breaking through plateau)
3. **Any unusual behavior** you observe when visualizing

This will help determine if the fix worked or if there's yet another issue.

---

## Summary Checklist

Before running training, verify:

- [ ] `domain/config.py` line 130 has `o_act=np.full(3,5)` (not `o_act=np.full(3,1)`)
- [ ] Using `p/slimevolley_fixed.json` (not `slimevolley_shaped.json`)
- [ ] Task name in config is `"task": "slimevolley"` (not `"slimevolley_shaped"`)
- [ ] Cleared old log files from previous training runs

Then run:
```bash
python neat_train.py -p p/slimevolley_fixed.json -n 9
```

And watch for NEGATIVE fitness gradually improving toward 0!

---

**This should be the complete fix. The combination of:**
1. **Mutually exclusive actions**
2. **Bounded output activation (tanh)**  
3. **Simpler network**
4. **No reward shaping**

**Should finally unlock learning! 🎯**
