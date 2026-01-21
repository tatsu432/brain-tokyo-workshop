# SlimeVolley Training Bottleneck Analysis

## Problem Summary
Training is stuck at fitness ~-3.33 (started at -4.33) after 1100+ generations. The agent (yellow slime) moves around the edge of the battlefield without meaningful gameplay.

---

## Root Causes Identified

### 🔴 CRITICAL: Action Mapping Issues

**Problem Location:** `domain/slimevolley.py` lines 103-141

The `_process_action()` function has problematic logic:

```python
# Current implementation (PROBLEMATIC):
if len(action) == 1:
    val = float(action[0])
    binary_action = np.array([
        1 if val > 0.33 else 0,   # forward
        1 if val < -0.33 else 0,  # backward  
        1 if abs(val) > 0.5 else 0 # jump
    ], dtype=np.int8)
else:
    # Multiple continuous values
    binary_action = np.array([
        1 if action[0] > 0 else 0,  # forward
        1 if action[1] > 0 else 0,  # backward
        1 if action[2] > 0 else 0   # jump
    ], dtype=np.int8)
```

**Why This Is Bad:**

1. **Conflicting Actions**: When NEAT outputs 3 values, both forward AND backward can be active simultaneously (if both > 0), which cancels out and wastes the network's capacity.

2. **Poor Threshold**: Using 0 as the threshold means random initialization likely activates many actions at once.

3. **Single Output Mode**: When network outputs 1 value, the mapping creates a limited action space that can't represent all 8 possible action combinations.

**Impact:** The agent likely gets stuck with conflicting actions, explaining the "moving around the edge" behavior.

---

### 🟡 MAJOR: Sparse Reward Structure

**Problem:** SlimeVolley only gives rewards when someone loses a life:
- +1 when opponent loses a life
- -1 when agent loses a life  
- 0 otherwise

**Current Performance:** -3.33 fitness means the agent loses ~3-4 lives per episode on average.

**Impact:** 
- Very sparse learning signal
- Hard to differentiate between "doing nothing" and "bad strategy"
- No reward shaping to encourage ball contact, positioning, etc.

---

### 🟡 MAJOR: Network Architecture Mismatch

**Current Config:**
```json
"layers": [15, 10]  // Hidden layers of size 15 and 10
"output_size": 3
```

**Problems:**
1. **Overcomplicated**: Two hidden layers with 15 and 10 nodes for a 12→3 mapping is excessive for early evolution
2. **Slow exploration**: Large networks have more parameters to evolve, slowing down initial learning

**Comparison:** SwingUp (successful) uses `[5, 5]` for 5→1 mapping, which is proportionally much simpler.

---

### 🟢 MINOR: Evaluation Settings

**Current:** `alg_nReps: 3` (only 3 trials per individual)

**Impact:** SlimeVolley is stochastic (opponent varies). 3 trials might not capture true fitness, leading to:
- High variance in fitness estimates
- Evolution might select "lucky" individuals over truly better ones

---

## Recommended Solutions (Priority Order)

### 1. ✅ FIX ACTION MAPPING (Highest Priority)

**Option A: Mutually Exclusive Forward/Backward (Recommended)**

Replace the action mapping with:

```python
def _process_action(self, action):
    """
    Map NEAT output to SlimeVolley actions with better logic.
    
    Actions: [forward, backward, jump]
    Network outputs 3 continuous values in range [-1, 1]
    """
    action = np.array(action).flatten()
    
    if len(action) == 1:
        # Single output: map to discrete movement + jump
        val = float(action[0])
        if val > 0.5:
            return np.array([1, 0, 1], dtype=np.int8)  # forward + jump
        elif val > 0:
            return np.array([1, 0, 0], dtype=np.int8)  # forward
        elif val < -0.5:
            return np.array([0, 1, 1], dtype=np.int8)  # backward + jump
        elif val < 0:
            return np.array([0, 1, 0], dtype=np.int8)  # backward
        else:
            return np.array([0, 0, 1], dtype=np.int8)  # jump only
    else:
        # 3 outputs: [horizontal_movement, jump, unused]
        # Make forward/backward mutually exclusive
        if action[0] > 0.2:
            forward = 1
            backward = 0
        elif action[0] < -0.2:
            forward = 0
            backward = 1
        else:
            forward = 0
            backward = 0
        
        jump = 1 if action[1] > 0.3 else 0
        
        return np.array([forward, backward, jump], dtype=np.int8)
```

**Why This Works:**
- Prevents conflicting forward/backward actions
- Clear deadband (-0.2 to 0.2) prevents random activation
- Higher threshold for jump (0.3) makes it a deliberate action
- Simpler mapping helps evolution find working strategies faster

---

### 2. ✅ SIMPLIFY NETWORK ARCHITECTURE

**New Config:**
```json
{
    "layers": [8, 8],  // Reduced from [15, 10]
    "alg_nReps": 5,    // Increased from 3
    "popSize": 256,    // Keep same
    "maxGen": 2048,
    "alg_act": 0,      // Allow all activation functions (was 5)
    // ... other params stay same
}
```

**Rationale:**
- Smaller network = faster evolution in early generations
- Can still evolve complexity as needed
- `alg_act: 0` lets NEAT try different activation functions

---

### 3. ✅ ADD REWARD SHAPING (Medium Priority)

Create a wrapper to add dense rewards:

```python
class SlimeVolleyRewardShapingEnv(SlimeVolleyEnv):
    """Add dense rewards to encourage learning"""
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        # Extract ball and agent positions from observation
        agent_x, agent_y = obs[0], obs[1]
        ball_x, ball_y = obs[4], obs[5]
        
        # Dense reward shaping
        shaped_reward = reward  # Start with original sparse reward
        
        # 1. Reward for being close to ball horizontally
        horizontal_dist = abs(agent_x - ball_x)
        if horizontal_dist < 0.3:
            shaped_reward += 0.1
        
        # 2. Penalty for being at edges (explains current behavior!)
        if abs(agent_x) > 0.8:  # Assuming normalized coords [-1, 1]
            shaped_reward -= 0.05
        
        # 3. Reward for ball height when agent is close
        if horizontal_dist < 0.4 and ball_y > agent_y:
            shaped_reward += 0.05
        
        return obs, shaped_reward, done, info
```

**Usage:**
```python
# In domain/slimevolley.py, modify the __init__ or add config option
```

---

### 4. ✅ INCREASE EVALUATION TRIALS

Change config:
```json
"alg_nReps": 8  // Increase from 3 to 8
```

**Trade-off:** Slower training but more reliable fitness estimates.

---

### 5. ✅ ADD CURRICULUM LEARNING (Optional, Advanced)

Start with easier opponents, gradually increase difficulty:

```python
class SlimeVolleyEasyCurriculum(SlimeVolleyEnv):
    """Curriculum learning: start with weaker opponents"""
    
    def __init__(self, difficulty=0.0):
        super().__init__()
        self.difficulty = difficulty  # 0.0 = easiest, 1.0 = hardest
        
    def reset(self):
        obs = super().reset()
        # Could modify opponent strength here if slimevolleygym supports it
        return obs
```

---

## Immediate Action Plan

### Phase 1: Quick Fixes (Do First)

1. **Fix action mapping** in `domain/slimevolley.py`
2. **Update config** to use simpler network:
   ```bash
   # Create new config file
   cp p/slimevolley.json p/slimevolley_fixed.json
   # Edit: layers=[8,8], alg_nReps=5, alg_act=0
   ```

3. **Test the changes:**
   ```bash
   python neat_train.py -p p/slimevolley_fixed.json -n 9
   ```

### Phase 2: Monitor Progress

**Expected improvements:**
- **First 50 gens:** Should see faster improvement than before
- **Gen 100-200:** Should break past -3.0 fitness
- **Gen 500+:** Should approach -1.0 or better (winning some points)

**If still stuck:**
- Add reward shaping (Phase 3)
- Consider reducing episode length to speed up learning
- Increase population diversity (lower `spec_thresh`)

### Phase 3: Advanced Improvements (If Needed)

1. Implement reward shaping wrapper
2. Add curriculum learning
3. Experiment with different activation functions
4. Try larger population with longer evolution

---

## Expected Outcomes

### Before Fixes:
- ✗ Stuck at -3.33 fitness
- ✗ Agent moves around edges
- ✗ No meaningful gameplay
- ✗ Conflicting actions waste network capacity

### After Fixes:
- ✓ Clearer action space → better exploration
- ✓ Simpler network → faster early learning
- ✓ Better action mapping → meaningful movements
- ✓ More reliable fitness → better evolution

### Timeline Estimate:
- **Immediate** (0-50 gens): Should see movement away from -3.33
- **Short-term** (100-300 gens): Should reach -2.0 to -1.5
- **Medium-term** (500-1000 gens): Should reach competitive play (-1.0 or better)

---

## Debugging Tools

### Check Current Behavior:

```python
# Test action mapping
from domain.slimevolley import SlimeVolleyEnv
env = SlimeVolleyEnv()

# Test various outputs
test_actions = [
    [0.5, 0.5, 0.5],   # All positive
    [-0.5, 0.5, 0.5],  # Mixed
    [0.1, -0.1, 0.1],  # Near zero
]

for action in test_actions:
    binary = env._process_action(action)
    print(f"Input: {action} → Output: {binary}")
```

### Visualize Best Individual:

```bash
# After some training, visualize the best agent
python test_slimevolley.py --view --gen 100
```

---

## Alternative Approaches (If Above Fails)

### 1. Single Output Network
Change to output 1 value and map to discrete actions:
- -1.0 to -0.6: backward
- -0.6 to -0.2: backward + jump  
- -0.2 to +0.2: jump only
- +0.2 to +0.6: forward
- +0.6 to +1.0: forward + jump

### 2. Different Architecture
Try a recurrent network (LSTM/GRU) to handle temporal patterns in ball movement.

### 3. Co-evolution
Evolve both agents simultaneously for more diverse training signal.

---

## Summary

**Main Issue:** Action mapping allows conflicting actions + network too complex

**Quick Fix:** 
1. Update `_process_action()` to prevent forward+backward simultaneously
2. Use simpler network: `layers=[8,8]`, `alg_act=0`
3. Increase `alg_nReps=5`

**Expected:** Should break through -3.33 plateau within 100-200 generations.

**Next Steps if Stuck:** Add reward shaping, reduce network complexity further, or try curriculum learning.
