# Understanding the `layers` Parameter in NEAT

## Key Finding: `layers` May Not Be Used for Initial Population!

After examining the NEAT source code, here's what I found:

### How NEAT Actually Initializes Networks

Looking at `neat_src/neat.py`, the `initPop()` function creates networks with:

1. **Input nodes**: 12 nodes (for slimevolley)
2. **Bias node**: 1 node
3. **Output nodes**: 6 nodes (for slimevolley)
4. **Hidden nodes**: **ZERO initially!**

The initial network structure is:
```
Inputs (12) → Outputs (6)
     ↑
   Bias (1)
```

**No hidden layers are created initially!**

### What About `layers=[8, 8]`?

The `layers` parameter is stored in hyperparameters as `ann_layers` with the comment:
```python
hyp["ann_layers"] = task.layers  # if fixed toplogy is used
```

However, I **don't see it being used** in the standard `initPop()` function. This suggests:

1. **It might be for a fixed-topology mode** (if that exists in this codebase)
2. **It might be legacy/unused** code
3. **It might be used elsewhere** (but I haven't found it)

### How Hidden Nodes Are Actually Added

Hidden nodes are added through **mutation** (`mutAddNode`), not during initialization:
- `prob_addNode: 0.08` means 8% chance per generation to add a hidden node
- When a node is added, it splits an existing connection
- Over time, networks grow from minimal to complex

### What This Means for Your Configuration

**Changing `layers=[8, 8]` to `layers=[15, 10]` might not actually do anything!**

The networks will still start minimal (inputs → outputs) and grow through mutation.

### Verification Needed

To confirm whether `layers` is actually used, you could:

1. **Check if there's a fixed-topology mode** in the codebase
2. **Test empirically**: Change `layers` and see if initial networks differ
3. **Search for other uses** of `ann_layers` in the codebase

### Recommendation

Since we're not sure if `layers` is used, the **other hyperparameter changes are more important**:

1. ✅ **Reward scale** (0.01 → 0.05) - **CRITICAL, definitely works**
2. ✅ **Evaluation reps** (5 → 10) - **CRITICAL, definitely works**  
3. ⚠️ **Network size** (`layers`) - **Uncertain if it has effect**

The reward scale and evaluation reps changes should have the biggest impact on training.

### If You Want to Ensure Larger Networks

If `layers` doesn't work, you can encourage network growth by:
- Increasing `prob_addNode` (already done: 0.05 → 0.08)
- Increasing `prob_addConn` (already done: 0.05 → 0.08)
- These mutations will add hidden nodes over time

The networks will evolve from minimal to larger topologies naturally through NEAT's mutation operators.
