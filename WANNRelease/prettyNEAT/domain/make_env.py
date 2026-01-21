import numpy as np
import gymnasium as gym
from matplotlib.pyplot import imread


def make_env(env_name, seed=-1, render_mode=False):
  # -- Bullet Environments ------------------------------------------- -- #
  if "Bullet" in env_name:
    import pybullet as p # pip install pybullet
    import pybullet_envs
    import pybullet_envs.bullet.kukaGymEnv as kukaGymEnv

  # -- Bipedal Walker ------------------------------------------------ -- #
  if (env_name.startswith("BipedalWalker")):
    if (env_name.startswith("BipedalWalkerHardcore")):
      import Box2D
      from domain.bipedal_walker import BipedalWalkerHardcore
      env = BipedalWalkerHardcore()
    elif (env_name.startswith("BipedalWalkerMedium")): 
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()
      env.accel = 3
    else:
      from domain.bipedal_walker import BipedalWalker
      env = BipedalWalker()


  # -- VAE Racing ---------------------------------------------------- -- #
  elif (env_name.startswith("VAERacing")):
    from domain.vae_racing import VAERacing
    env = VAERacing()
    
    
  # -- Classification ------------------------------------------------ -- #
  elif (env_name.startswith("Classify")):
    from domain.classify_gym import ClassifyEnv
    if env_name.endswith("digits"):
      from domain.classify_gym import digit_raw
      trainSet, target  = digit_raw()
        
    if env_name.endswith("mnist256"):
      from domain.classify_gym import mnist_256
      trainSet, target  = mnist_256()

    env = ClassifyEnv(trainSet,target)  


  # -- Cart Pole Swing up -------------------------------------------- -- #
  elif (env_name.startswith("CartPoleSwingUp")):
    from domain.cartpole_swingup import CartPoleSwingUpEnv
    env = CartPoleSwingUpEnv()
    if (env_name.startswith("CartPoleSwingUp_Hard")):
      env.dt = 0.01
      env.t_limit = 200

  elif env_name == 'SlimeVolley-Shaped-v0':
      from domain.slimevolley_shaped import SlimeVolleyShapedEnv
      return SlimeVolleyShapedEnv()

  # -- Slime Volleyball ---------------------------------------------- -- #
  elif (env_name.startswith("SlimeVolley")):
    # Support different variants:
    # - SlimeVolley-v0: Standard sparse rewards
    # - SlimeVolley-Shaped-v0: Dense reward shaping (easier to learn)
    if "Shaped" in env_name:
      from domain.slimevolley import SlimeVolleyRewardShapingEnv
      # shaping_weight controls how much shaped rewards matter
      # 0.01 = small nudges, 0.1 = significant influence
      env = SlimeVolleyRewardShapingEnv(shaping_weight=0.01)
    else:
      from domain.slimevolley import SlimeVolleyEnv
      env = SlimeVolleyEnv()

  # -- Other  -------------------------------------------------------- -- #
  else:
    env = gym.make(env_name)

  if (seed >= 0):
    env.seed(seed)

  return env