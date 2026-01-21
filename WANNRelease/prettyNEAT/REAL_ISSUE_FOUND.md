# 🔴 REAL ISSUE FOUND: Output Activation Function is Unbounded!

## The Smoking Gun

I found the root cause! Look at `domain/config.py` line 130:

```python
slimevolley = Game(
    ...
    o_act=np.full(3,1),  # ← THIS IS THE PROBLEM!
    ...
)
```

### What `o_act=1` means:
- Output nodes use activation function #1 = **LINEAR**
- Linear activation: `f(x) = x` (no bounding!)
- Output range: **[-∞, +∞]** (unbounded!)

### Why this breaks everything:

Your action mapping checks thresholds like:
```python
if action[0] > 0.2:  # forward
    forward = 1
elif action[0] < -0.2:  # backward
    backward = 1
```

But with linear activation, the network can output:
- `[500.0, 300.0, 200.0]` → ALL actions trigger!
- `[-500.0, -300.0, -200.0]` → All negative, backward always triggers
- ANY large value breaks the threshold logic

### Comparison with SwingUp (which works):

```python
# From config.py - CartPole SwingUp
cartpole_swingup = Game(
    ...
    o_act=np.full(1,1),  # Also linear!
    ...
)
```

**But SwingUp only has 1 output** (force direction), so linear is fine. The environment normalizes it.

**SlimeVolley has 3 outputs** that you're thresholding - you NEED bounded outputs!

---

## The Fix

### Option 1: Use Tanh Output Activation (Recommended)

```python
# In domain/config.py
slimevolley = Game(
    ...
    o_act=np.full(3, 5),  # 5 = tanh, outputs in [-1, 1]
    ...
)
```

**Why tanh?**
- Outputs bounded to [-1, 1]
- Symmetric around 0 (good for forward/backward)
- Your action mapping thresholds (0.2, 0.3) make sense in this range

### Option 2: Use Sigmoid Output Activation

```python
o_act=np.full(3, 6),  # 6 = sigmoid, outputs in [0, 1]
```

But then you'd need to update action mapping thresholds for [0,  1] range.

---

## Why Reward Shaping Showed +5 Fitness

The positive fitness (+3 to +5.57) proves **reward hacking**:

1. Original SlimeVolley: ~-3 to -4 (agent losing)
2. Shaped rewards per step: ~1-2 points
3. Over 3000 steps: +3000 to +6000 shaped reward
4. With weight=0.01: +30 to +60 total shaped
5. Combined: -4 + 40 = +36... wait that's too high

Actually, let me recalculate. Looking at the shaped reward code:
- Per step shaped reward: ~0.5 to 3.0  
- Over 3000 steps: ~1500 to 9000
- With weight=0.01: ~15 to 90
- Original reward: -3 to -5
- **Total: +10 to +85**

But you're seeing +3 to +5.5, which suggests:
- Maybe not all steps get rewards
- Or the shaped rewards average out
- But still dominating the signal!

---

## Action Plan (UPDATED)

### IMMEDIATE: Fix Output Activation

1. Edit `domain/config.py` line 130:
   ```python
   o_act=np.full(3, 5),  # Change from 1 (linear) to 5 (tanh)
   ```

2. Test with the BASIC fixed version (no shaping):
   ```bash
   python neat_train.py -p p/slimevolley_fixed.json -n 9
   ```

3. Monitor fitness:
   - Should be **NEGATIVE** (-4 to -3 initially)
   - Should gradually approach 0 and then go positive
   - Positive early = still reward hacking

### If Still Stuck

There may be another issue with how NEAT uses the output activation. Let me investigate further.

---

## Activation Function Reference

From `neat_src/ann.py`:

| ID | Name | Formula | Output Range |
|----|------|---------|--------------|
| 1 | Linear | `x` | [-∞, +∞] **← PROBLEM!** |
| 2 | Step | `1 if x>0 else 0` | {0, 1} |
| 3 | Sin | `sin(πx)` | [-1, 1] |
| 4 | Gaussian | `exp(-x²/2)` | [0, 1] |
| 5 | Tanh | `tanh(x)` | [-1, 1] **← USE THIS** |
| 6 | Sigmoid | `(tanh(x/2)+1)/2` | [0, 1] |
| 7 | Inverse | `-x` | [-∞, +∞] |
| 8 | Abs | `|x|` | [0, +∞] |
| 9 | ReLU | `max(0,x)` | [0, +∞] |
| 10 | Cos | `cos(πx)` | [-1, 1] |

**For action mapping to work, you NEED bounded outputs (3, 4, 5, 6, or 10).**

---

## Expected Results After Fix

### Before (with linear output):
```
Network outputs: [50.0, 30.0, 20.0] (unbounded!)
Action mapping: [1, 1, 1] (all trigger - conflict!)
```

### After (with tanh output):
```
Network outputs: [0.8, -0.3, 0.5] (bounded to [-1,1])
Action mapping: [1, 0, 1] (forward + jump - correct!)
```

This should finally allow the agent to learn!
