# Quick Start: Fixed SlimeVolley Training

## TL;DR - What to Do Now

Your training was stuck at **-3.33 fitness** due to a bug where the agent could activate forward+backward simultaneously. This has been fixed.

**Run this command to start training with the fix:**

```bash
cd /Users/a81808/code/self_study/brain-tokyo-workshop/WANNRelease/prettyNEAT
python neat_train.py -p p/slimevolley_fixed.json -n 9
```

**Expected:** Fitness should improve past -3.33 within 100-200 generations.

---

## What Was Fixed

### 🔴 Critical Fix: Action Mapping

**Problem:** Agent could activate forward AND backward at the same time, causing actions to cancel out.

**Solution:** Made forward/backward mutually exclusive with a deadband.

**Verification:**
```bash
python test_action_mapping.py
# Should show: ✓ All 16 tests passed!
```

### 🟡 Important Improvements

1. **Simpler network:** `[15,10]` → `[8,8]` layers (faster early learning)
2. **More trials:** 3 → 5 evaluations per individual (more reliable)
3. **Better exploration:** Allow all activation functions instead of just tanh

---

## Training Options

### Option 1: Standard Fixed Version (Recommended)

```bash
python neat_train.py -p p/slimevolley_fixed.json -n 9
```

**Best for:** Most cases - cleaner learning signal

### Option 2: With Reward Shaping (If Option 1 Stalls)

```bash
python neat_train.py -p p/slimevolley_shaped.json -n 9
```

**Best for:** If stuck again after 200 generations with Option 1

**What it does:** Adds dense rewards for:
- Moving toward ball (instead of edges)
- Good positioning
- Ball engagement

---

## How to Monitor Progress

### Good Signs ✅

```
Gen    Elite Fit    Best Fit    What it means
0      -4.33        -4.33       Starting point
50     -3.67        -3.67       Improving
100    -3.00        -3.00       ← BREAKING THROUGH! 🎉
200    -2.33        -2.33       Continued progress
500    -1.33        -1.33       Competitive play
```

### Bad Signs ❌

```
Gen    Elite Fit    Best Fit    What it means
0      -4.33        -4.33       Starting point
100    -3.33        -3.33       ← Still stuck
200    -3.33        -3.33       Not improving
```

**If you see this:** Try Option 2 (reward shaping) or see Troubleshooting below.

---

## What Changed in the Code

### Files Modified

- ✅ `domain/slimevolley.py` - Fixed action mapping
- ✅ `domain/config.py` - Simplified network (15,10 → 8,8)
- ✅ `domain/make_env.py` - Added reward shaping support

### New Files Created

- ✅ `p/slimevolley_fixed.json` - Recommended config
- ✅ `p/slimevolley_shaped.json` - With reward shaping
- ✅ `test_action_mapping.py` - Verification tests
- ✅ `SLIMEVOLLEY_DIAGNOSIS.md` - Detailed analysis
- ✅ `FIXES_APPLIED.md` - Implementation details
- ✅ `BEFORE_AFTER_COMPARISON.md` - Side-by-side comparison
- ✅ `QUICK_START_FIXED.md` - This file

---

## Troubleshooting

### Still Stuck at -3.33 After 200 Generations?

**Try reward shaping:**
```bash
python neat_train.py -p p/slimevolley_shaped.json -n 9
```

### Training Too Slow?

**Reduce workers or population:**
```bash
# Fewer workers (faster but less parallel)
python neat_train.py -p p/slimevolley_fixed.json -n 5

# Or edit p/slimevolley_fixed.json:
# "popSize": 128  (was 256)
```

### Want to Understand More?

**Read the documentation:**
1. `FIXES_APPLIED.md` - What changed and why
2. `SLIMEVOLLEY_DIAGNOSIS.md` - Deep dive analysis
3. `BEFORE_AFTER_COMPARISON.md` - Side-by-side comparison

**Run the tests:**
```bash
python test_action_mapping.py
```

---

## Expected Timeline

| Generations | Fitness Range | Agent Behavior |
|-------------|---------------|----------------|
| 0-50 | -4.0 to -3.5 | Initial exploration |
| 50-200 | -3.5 to -2.5 | **Breaking plateau**, ball engagement |
| 200-500 | -2.5 to -1.5 | Competitive play |
| 500-1000 | -1.5 to -0.5 | Winning points consistently |
| 1000+ | -0.5 to +2.0 | Strong player |

---

## The Bug Explained (Simple Version)

**What happened:**
- Your network could output: `[forward=0.5, backward=0.5, jump=0.5]`
- Old code translated this to: `[forward=1, backward=1, jump=1]`
- Game engine: "Forward AND backward? That cancels out!"
- Result: Agent doesn't move, just wiggles in place

**The fix:**
- Network outputs: `[horizontal=0.5, jump=0.5, unused=0.0]`
- New code: "horizontal=0.5 > 0.2, so forward=1, backward=0"
- Game engine: "Forward only? Got it!"
- Result: Agent moves forward (as intended)

**Why it caused the plateau:**
- Evolution tried to learn good strategies
- But the action space was fundamentally broken
- Like trying to navigate with a compass that points random directions
- Agent learned "stay near edges" as the best bad strategy
- Fitness stuck at -3.33 (losing most points)

**Why the fix works:**
- Clear action space → Meaningful movements
- Meaningful movements → Can reach ball
- Reaching ball → Can win points
- Winning points → Fitness improves ✅

---

## Next Steps

1. **Start training:** `python neat_train.py -p p/slimevolley_fixed.json -n 9`
2. **Watch progress:** Should see fitness > -3.33 within 100 gens
3. **If successful:** Let it run for 1000+ generations
4. **If stuck:** Try reward shaping version
5. **Visualize results:** Use the rendering tools to watch your agent play!

---

## Questions?

- **"How do I know if it's working?"** - Fitness should improve past -3.33 within 100-200 gens
- **"What if it's still stuck?"** - Try reward shaping version or see SLIMEVOLLEY_DIAGNOSIS.md
- **"Can I continue my old training?"** - Not recommended, population learned wrong action space
- **"How long will it take?"** - Expect 500-1000 generations for competitive play

---

**That's it! The main issue was conflicting actions. The fix should get you past the -3.33 plateau. Good luck! 🚀**
