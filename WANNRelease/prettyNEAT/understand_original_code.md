# Understanding the ORIGINAL Code

## Original Action Mapping

```python
# ORIGINAL (from git commit 14d93f0):
binary_action = np.array([
    1 if action[0] > 0 else 0,  # forward
    1 if action[1] > 0 else 0,  # backward
    1 if action[2] > 0 else 0   # jump
], dtype=np.int8)
```

### The Bug in Original Code:

**Network outputs:** `[0.5, 0.3, 0.2]` (3 independent values)

**Action mapping:**
- `action[0] = 0.5 > 0` → forward = 1 ✓
- `action[1] = 0.3 > 0` → backward = 1 ✓
- `action[2] = 0.2 > 0` → jump = 1 ✓

**Result:** `[1, 1, 1]` - Forward AND Backward AND Jump!

**Problem:** SlimeVolley interprets [1, 1, 0] as "go forward AND backward simultaneously"
→ These actions CANCEL OUT in the physics engine
→ Agent doesn't move horizontally, just wiggles in place

---

## My "Fix" (Which Made Things Worse)

### V1 Fix: Mutually Exclusive Actions
```python
# Made forward/backward mutually exclusive
if action[0] > 0.2: forward=1, backward=0
elif action[0] < -0.2: forward=0, backward=1
```

**Issue:** Now using action[0] for BOTH forward and backward
→ Only 2 outputs needed: [horizontal, jump]
→ But network still has 3 outputs!
→ What happens to action[2]? It's ignored!

### V2 Fix: Added Tanh Activation
```python
# In domain/config.py
o_act=np.full(3, 5),  # Tanh activation
```

**Issue:** Tanh outputs centered around 0
→ Thresholds (0.2, 0.3) too high
→ 34% of outputs in deadband
→ Agent rarely acts
→ Fitness WORSE (-4.60 vs -3.33)

---

## The REAL Issue I Just Realized

### Configuration Says output_size=3

From `domain/config.py`:
```python
slimevolley = Game(
    output_size=3,  # THREE outputs
    o_act=np.full(3, 5),  # All use tanh
)
```

### But My Action Mapping Uses Only 2!

```python
# My current code:
if action[0] > 0.05: forward=1  # Uses action[0]
elif action[0] < -0.05: backward=1  # Uses action[0]
jump = 1 if action[1] > 0.1 else 0  # Uses action[1]
# action[2] is IGNORED!
```

**This might be the problem!**

The network has 3 outputs, but I'm only using 2 of them. This could confuse evolution because changing the third output has no effect.

---

## Two Possible Approaches

### Approach A: Keep 3 Outputs, Use All 3

```python
# Use all three network outputs:
forward = 1 if action[0] > 0.05 else 0
backward = 1 if action[1] > 0.05 else 0
jump = 1 if action[2] > 0.1 else 0

# Resolve conflict if both forward and backward activated:
if forward and backward:
    # Tie-breaking: use whichever has higher magnitude
    if abs(action[0]) > abs(action[1]):
        backward = 0
    else:
        forward = 0
```

### Approach B: Change to 2 Outputs

```python
# In domain/config.py:
output_size=2,  # Only [horizontal, jump]

# In action mapping:
if action[0] > 0.05: forward=1, backward=0
elif action[0] < -0.05: forward=0, backward=1
else: forward=0, backward=0
jump = 1 if action[1] > 0.1 else 0
```

---

## Which is Better?

**Approach A (3 outputs, resolve conflicts):**
- Pro: More expressive (network can learn forward and backward independently)
- Pro: No config change needed
- Con: More complex to evolve (3D action space)
- Con: Conflict resolution might confuse learning

**Approach B (2 outputs, combined horizontal):**
- Pro: Simpler action space (2D)
- Pro: No conflicts possible
- Pro: Clear gradient (horizontal value directly controls direction)
- Con: Requires config change
- Con: May be less expressive

**I think Approach B is better!** It's cleaner and matches the actual control scheme better.

---

## But Wait - Check What SwingUp Does

SwingUp has `output_size=1` and uses linear activation, and it WORKS.

How does SwingUp's action mapping work? Let me check...

Actually, SwingUp might just pass the output directly to the environment without any mapping, and the environment handles it.

For SlimeVolley, we NEED to map to MultiBinary(3), so we need explicit mapping logic.

---

## My Recommendation: Switch to 2 Outputs

1. Change `output_size` from 3 to 2
2. Keep current action mapping (horizontal + jump)
3. Lower thresholds to match tanh (0.05, 0.1)
4. This should work!
