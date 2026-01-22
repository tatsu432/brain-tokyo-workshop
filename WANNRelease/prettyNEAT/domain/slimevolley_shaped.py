"""
SlimeVolley with Dense Reward Shaping for NEAT Training

The problem: SlimeVolley's sparse reward (+1/-1 only when points are scored)
makes it very hard for NEAT to learn because:
1. Random policies always lose 5-0 or 5-1
2. There's no gradient between "random" and "slightly better than random"
3. Selection can't distinguish networks that all get -4 or -5

Solution: Add dense reward shaping with decomposed fitness:
- Ball touches (primary skill signal)
- Rallies won (game progress)
- Ball time on opponent side (offensive pressure)
- Avoid self-goals heavily

The shaped rewards are carefully weighted so that the final game outcome
dominates once the agent starts scoring, avoiding reward hacking.

Key Design Principles for NEAT:
1. Early learning signal: Small rewards for basic behaviors (moving, touching ball)
2. Progressive rewards: Hitting > Touching > Tracking
3. Game outcome dominance: Final score weighted heavily to prevent gaming
4. No gradient reliance: Works with evolutionary selection

USAGE:
1. Copy this file to domain/slimevolley_shaped.py
2. Register it in domain/make_env.py
3. Use 'SlimeVolley-Shaped-v0' as env_name in config.py
"""

import logging
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random

logger = logging.getLogger(__name__)

try:
    import slimevolleygym
    from slimevolleygym.slimevolley import SlimeVolleyEnv as BaseSlimeVolleyEnv
    import gym as old_gym
    SLIMEVOLLEY_AVAILABLE = True
except ImportError:
    SLIMEVOLLEY_AVAILABLE = False
    old_gym = None
    BaseSlimeVolleyEnv = None


class SlimeVolleyShapedEnv(gym.Env):
    """
    SlimeVolley with dense reward shaping optimized for NEAT/evolutionary methods.
    
    Decomposed Fitness Components (configurable via constructor):
    
    1. Ball Touches: +touch_bonus when agent hits the ball
       - Primary skill signal for early learning
       - Detects velocity changes when ball is near agent
    
    2. Rallies Won: +rally_bonus for each point scored
       - Directly tied to winning
       - Scaled to dominate once agent learns to score
    
    3. Ball Position: +ball_opponent_side_bonus * time_ratio when ball on opponent's side
       - Encourages offensive play
       - Small continuous reward for maintaining pressure
    
    4. Self-Goals Penalty: -self_goal_penalty for losing points
       - Discourages risky play that leads to losing
       - Weighted to make net positive play beneficial
    
    5. Movement/Tracking (small): Tiny rewards for moving toward ball
       - Helps initial random search find "ball-aware" networks
       - Minimal weight to not dominate once hitting is learned
    
    Final episode fitness combines all components with heavy weight on
    the actual game outcome (rallies_won - self_goals) to ensure
    the agent optimizes for winning, not reward hacking.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        # Shaping weights - tune these for your training curriculum
        touch_bonus: float = 0.1,           # Reward per ball touch
        rally_bonus: float = 1.0,           # Reward per rally/point won  
        self_goal_penalty: float = 0.5,     # Penalty per point lost
        ball_opponent_side_bonus: float = 0.01,  # Per-step bonus when ball on opponent side
        tracking_bonus: float = 0.02,       # Small bonus for moving toward ball
        game_outcome_weight: float = 10.0,  # Final multiplier for net game score
        shaping_scale: float = 1.0,         # Global scale for all shaping rewards
    ):
        """
        Args:
            touch_bonus: Reward for each ball touch (detected via velocity change)
            rally_bonus: Reward for winning a rally (scoring a point)
            self_goal_penalty: Penalty for losing a rally
            ball_opponent_side_bonus: Small per-step reward when ball is on opponent's side
            tracking_bonus: Small reward for moving toward the ball
            game_outcome_weight: Weight multiplier for final game outcome (wins - losses)
            shaping_scale: Global multiplier for all shaping rewards (set 0 to disable)
        """
        if not SLIMEVOLLEY_AVAILABLE:
            raise ImportError("slimevolleygym not installed")
        
        # Use direct instantiation instead of gym.make() to support self-play
        # gym.make() wraps with OrderEnforcing which doesn't pass through otherAction
        self.env = BaseSlimeVolleyEnv()
        
        # Reward configuration
        self.touch_bonus = touch_bonus
        self.rally_bonus = rally_bonus
        self.self_goal_penalty = self_goal_penalty
        self.ball_opponent_side_bonus = ball_opponent_side_bonus
        self.tracking_bonus = tracking_bonus
        self.game_outcome_weight = game_outcome_weight
        self.shaping_scale = shaping_scale
        
        self.max_steps = 3000
        self.t = 0
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.action_space = spaces.MultiBinary(3)
        
        # State for reward shaping
        self.prev_obs = None
        
        # Episode statistics for fitness calculation
        self.episode_stats = {
            'ball_touches': 0,
            'rallies_won': 0,
            'rallies_lost': 0,
            'ball_time_opponent_side': 0,
            'total_steps': 0,
            'tracking_reward': 0.0,
            'raw_game_reward': 0.0,  # Cumulative actual game reward (-5 to +5)
        }
        
        self.np_random = None
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        return [seed]
    
    # Action mapping for 6 discrete actions:
    # Index 0: left + no jump  → [forward=0, backward=1, jump=0]
    # Index 1: stay + no jump  → [forward=0, backward=0, jump=0]
    # Index 2: right + no jump → [forward=1, backward=0, jump=0]
    # Index 3: left + jump     → [forward=0, backward=1, jump=1]
    # Index 4: stay + jump     → [forward=0, backward=0, jump=1]
    # Index 5: right + jump    → [forward=1, backward=0, jump=1]
    DISCRETE_ACTION_MAP = [
        [0, 1, 0],  # 0: left + no jump
        [0, 0, 0],  # 1: stay + no jump
        [1, 0, 0],  # 2: right + no jump
        [0, 1, 1],  # 3: left + jump
        [0, 0, 1],  # 4: stay + jump
        [1, 0, 1],  # 5: right + jump
    ]
    
    def _process_action(self, action):
        """Convert NEAT output to binary actions.
        
        Supports two modes:
        1. Discrete mode (int): action is an index 0-5 from probabilistic selection
        2. Continuous mode (array): action is continuous values with thresholds
        """
        # Handle discrete action index (from 'prob' actionSelect)
        if isinstance(action, (int, np.integer)):
            action_idx = int(action) % 6  # Ensure valid index
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)
        
        # Handle continuous action (legacy mode or different actionSelect)
        action = np.array(action).flatten()
        
        # If 6 outputs (discrete mode but passed as array), take argmax
        if len(action) == 6:
            action_idx = np.argmax(action)
            return np.array(self.DISCRETE_ACTION_MAP[action_idx], dtype=np.int8)
        
        # Legacy 2-output continuous mode with thresholds
        action = np.tanh(action)  # Ensure bounded
        
        if len(action) >= 2:
            if action[0] > 0.3:
                forward, backward = 1, 0
            elif action[0] < -0.3:
                forward, backward = 0, 1
            else:
                forward, backward = 0, 0
            jump = 1 if action[1] > 0.3 else 0
        else:
            val = float(action[0]) if len(action) > 0 else 0
            if val > 0.5:
                forward, backward, jump = 1, 0, 1
            elif val > 0.1:
                forward, backward, jump = 1, 0, 0
            elif val > -0.1:
                forward, backward, jump = 0, 0, 1
            elif val > -0.5:
                forward, backward, jump = 0, 1, 0
            else:
                forward, backward, jump = 0, 1, 1
        
        return np.array([forward, backward, jump], dtype=np.int8)
    
    def _detect_ball_touch(self, obs, prev_obs):
        """
        Detect if agent touched the ball using velocity change detection.
        
        Returns True if:
        1. Ball was on our side (ball_x < 0)
        2. Ball velocity changed significantly
        3. Agent was close to ball
        """
        if prev_obs is None:
            return False
        
        # Extract observations (scaled by 10 in slimevolleygym)
        agent_x = obs[0]
        agent_y = obs[1]
        ball_x = obs[4]
        ball_y = obs[5]
        ball_vx = obs[6]
        ball_vy = obs[7]
        
        prev_ball_x = prev_obs[4]
        prev_ball_vx = prev_obs[6]
        prev_ball_vy = prev_obs[7]
        
        # Ball must have been on our side
        if prev_ball_x >= 0:
            return False
        
        # Check for velocity change
        vx_change = abs(ball_vx - prev_ball_vx)
        vy_change = abs(ball_vy - prev_ball_vy)
        
        # Velocity flipped up (vy went from negative to positive) = definite hit
        vy_flipped = (prev_ball_vy < -0.1) and (ball_vy > 0.1)
        significant_change = (vx_change > 0.3) or (vy_change > 0.3)
        
        if not (vy_flipped or significant_change):
            return False
        
        # Agent must be close to ball for it to be our hit
        dist_x = abs(agent_x - ball_x)
        dist_y = abs(agent_y - ball_y)
        dist_2d = np.sqrt(dist_x**2 + dist_y**2)
        
        # Hit threshold: roughly agent_radius + ball_radius (scaled)
        HIT_THRESHOLD = 0.5
        
        return dist_2d < HIT_THRESHOLD
    
    def _compute_shaping_reward(self, obs, prev_obs, game_reward):
        """
        Compute decomposed shaping reward for NEAT fitness.
        
        Observation format (all values divided by scaleFactor=10.0):
        [agent_x, agent_y, agent_vx, agent_vy,
         ball_x, ball_y, ball_vx, ball_vy,
         opp_x, opp_y, opp_vx, opp_vy]
        
        Key: ball_x < 0 means ball is on OUR side
        """
        shaped_reward = 0.0
        
        # =================================================================
        # 1. BALL TOUCH DETECTION (most important for early learning)
        # =================================================================
        if self._detect_ball_touch(obs, prev_obs):
            shaped_reward += self.touch_bonus
            self.episode_stats['ball_touches'] += 1
        
        # =================================================================
        # 2. RALLY OUTCOME (from game reward)
        # =================================================================
        if game_reward > 0:  # We scored!
            shaped_reward += self.rally_bonus
            self.episode_stats['rallies_won'] += 1
        elif game_reward < 0:  # We lost a point
            shaped_reward -= self.self_goal_penalty
            self.episode_stats['rallies_lost'] += 1
        
        # =================================================================
        # 3. BALL POSITION PRESSURE
        # =================================================================
        ball_x = obs[4]
        if ball_x > 0:  # Ball on opponent's side
            shaped_reward += self.ball_opponent_side_bonus
            self.episode_stats['ball_time_opponent_side'] += 1
        
        # =================================================================
        # 4. TRACKING/ANTICIPATION (small reward for moving toward ball)
        # =================================================================
        if prev_obs is not None:
            agent_x = obs[0]
            ball_x = obs[4]
            prev_agent_x = prev_obs[0]
            
            ball_on_our_side = ball_x < 0
            
            if ball_on_our_side:
                # Distance to ball improved?
                prev_dist = abs(prev_agent_x - ball_x)
                curr_dist = abs(agent_x - ball_x)
                improvement = prev_dist - curr_dist
                
                if improvement > 0:
                    # Small bonus, capped
                    tracking = self.tracking_bonus * min(improvement / 0.1, 1.0)
                    shaped_reward += tracking
                    self.episode_stats['tracking_reward'] += tracking
        
        self.episode_stats['total_steps'] += 1
        
        return shaped_reward * self.shaping_scale
    
    def step(self, action, otherAction=None):
        """
        Take a step in the environment.
        
        Args:
            action: Action for our agent (right side)
            otherAction: Optional action for opponent (for self-play)
                        If None, uses built-in baseline policy
        """
        binary_action = self._process_action(action)
        
        prev_obs = self.prev_obs
        
        # Step with optional opponent action (for self-play)
        if otherAction is not None:
            other_binary = self._process_action(otherAction)
            obs, game_reward, done, info = self.env.step(binary_action, other_binary)
        else:
            obs, game_reward, done, info = self.env.step(binary_action)
        
        # Track raw game reward (actual score, not shaped)
        self.episode_stats['raw_game_reward'] += game_reward
        
        # Compute shaped reward
        total_reward = game_reward  # Start with actual game reward
        if self.shaping_scale > 0:
            shaping = self._compute_shaping_reward(obs, prev_obs, game_reward)
            total_reward += shaping
            info['shaping_reward'] = shaping
        
        # Store episode stats in info for analysis
        info['episode_stats'] = self.episode_stats.copy()
        info['game_reward'] = game_reward
        info['raw_game_reward'] = self.episode_stats['raw_game_reward']
        
        self.prev_obs = obs.copy()
        self.t += 1
        
        if self.t >= self.max_steps:
            done = True
        
        # At episode end, compute final fitness with game outcome weight
        if done:
            final_bonus = self.game_outcome_weight * (
                self.episode_stats['rallies_won'] - 
                self.episode_stats['rallies_lost']
            )
            total_reward += final_bonus
            info['final_game_bonus'] = final_bonus
        
        return obs, total_reward, done, info
    
    def reset(self):
        self.t = 0
        self.prev_obs = None
        
        # Reset episode statistics
        self.episode_stats = {
            'ball_touches': 0,
            'rallies_won': 0,
            'rallies_lost': 0,
            'ball_time_opponent_side': 0,
            'total_steps': 0,
            'tracking_reward': 0.0,
            'raw_game_reward': 0.0,
        }
        
        obs = self.env.reset()
        self.prev_obs = obs.copy()
        return obs
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def close(self):
        if self.env:
            self.env.close()
    
    def get_episode_stats(self):
        """Return current episode statistics."""
        return self.episode_stats.copy()


class SlimeVolleyCurriculumEnv(SlimeVolleyShapedEnv):
    """
    SlimeVolley with curriculum learning support.
    
    Curriculum stages:
    1. TOUCH: Focus on hitting the ball (high touch_bonus)
    2. RALLY: Focus on keeping rallies going (moderate all rewards)
    3. WIN: Focus on winning games (high game_outcome_weight)
    
    Call set_curriculum_stage() to switch between stages.
    """
    
    # Curriculum stage configurations
    CURRICULUM_CONFIGS = {
        'touch': {
            'touch_bonus': 0.3,
            'rally_bonus': 0.2,
            'self_goal_penalty': 0.1,
            'ball_opponent_side_bonus': 0.02,
            'tracking_bonus': 0.05,
            'game_outcome_weight': 2.0,
        },
        'rally': {
            'touch_bonus': 0.15,
            'rally_bonus': 0.5,
            'self_goal_penalty': 0.3,
            'ball_opponent_side_bonus': 0.01,
            'tracking_bonus': 0.02,
            'game_outcome_weight': 5.0,
        },
        'win': {
            'touch_bonus': 0.05,
            'rally_bonus': 1.0,
            'self_goal_penalty': 0.5,
            'ball_opponent_side_bonus': 0.005,
            'tracking_bonus': 0.01,
            'game_outcome_weight': 10.0,
        },
    }
    
    def __init__(self, initial_stage: str = 'touch', shaping_scale: float = 1.0):
        """
        Args:
            initial_stage: Starting curriculum stage ('touch', 'rally', or 'win')
            shaping_scale: Global scale for shaping rewards
        """
        # Initialize with touch stage defaults
        config = self.CURRICULUM_CONFIGS[initial_stage]
        super().__init__(
            shaping_scale=shaping_scale,
            **config
        )
        self.current_stage = initial_stage
    
    def set_curriculum_stage(self, stage: str):
        """Switch to a different curriculum stage."""
        if stage not in self.CURRICULUM_CONFIGS:
            raise ValueError(f"Unknown stage: {stage}. Use 'touch', 'rally', or 'win'")
        
        config = self.CURRICULUM_CONFIGS[stage]
        self.touch_bonus = config['touch_bonus']
        self.rally_bonus = config['rally_bonus']
        self.self_goal_penalty = config['self_goal_penalty']
        self.ball_opponent_side_bonus = config['ball_opponent_side_bonus']
        self.tracking_bonus = config['tracking_bonus']
        self.game_outcome_weight = config['game_outcome_weight']
        self.current_stage = stage


# Backward compatibility aliases
SlimeVolleyStrongShapedEnv = SlimeVolleyShapedEnv  # Use default params


if __name__ == "__main__":
    # Test the shaped environment
    print("Testing SlimeVolleyShapedEnv...")
    
    env = SlimeVolleyShapedEnv()
    
    # Test with random actions
    obs = env.reset()
    total_reward = 0
    total_shaping = 0
    
    for _ in range(1000):
        action = np.random.randint(0, 6)  # Random discrete action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if 'shaping_reward' in info:
            total_shaping += info['shaping_reward']
        if done:
            break
    
    stats = env.get_episode_stats()
    print(f"Total reward: {total_reward:.2f}")
    print(f"Shaping component: {total_shaping:.2f}")
    print(f"Episode stats: {stats}")
    
    env.close()
    
    # Test curriculum env
    print("\nTesting SlimeVolleyCurriculumEnv...")
    env = SlimeVolleyCurriculumEnv(initial_stage='touch')
    print(f"Initial stage: {env.current_stage}")
    
    env.set_curriculum_stage('rally')
    print(f"Changed to: {env.current_stage}")
    
    env.set_curriculum_stage('win')
    print(f"Changed to: {env.current_stage}")
    
    env.close()
    print("Test complete!")
