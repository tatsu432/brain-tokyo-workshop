import logging
import warnings

import gymnasium as gym

logger = logging.getLogger(__name__)

# Suppress gym step API deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*Initializing environment in old step API.*",
)
# Suppress render mode deprecation warning
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=".*The argument mode in render method is deprecated.*",
)


def make_env(env_name: str, seed: int = -1, render_mode: bool = False) -> gym.Env:
    # -- Bullet Environments ------------------------------------------- -- #
    if "Bullet" in env_name:
        pass

    # -- Bipedal Walker ------------------------------------------------ -- #
    if env_name.startswith("BipedalWalker"):
        if env_name.startswith("BipedalWalkerHardcore"):
            from domain.bipedal_walker import BipedalWalkerHardcore

            env = BipedalWalkerHardcore()
        elif env_name.startswith("BipedalWalkerMedium"):
            from domain.bipedal_walker import BipedalWalker

            env = BipedalWalker()
            env.accel = 3
        else:
            from domain.bipedal_walker import BipedalWalker

            env = BipedalWalker()

    # -- VAE Racing ---------------------------------------------------- -- #
    elif env_name.startswith("VAERacing"):
        from domain.vae_racing import VAERacing

        env = VAERacing()

    # -- Classification ------------------------------------------------ -- #
    elif env_name.startswith("Classify"):
        from domain.classify_gym import ClassifyEnv

        if env_name.endswith("digits"):
            from domain.classify_gym import digit_raw

            trainSet, target = digit_raw()

        if env_name.endswith("mnist256"):
            from domain.classify_gym import mnist_256

            trainSet, target = mnist_256()

        env = ClassifyEnv(trainSet, target)

    # -- Cart Pole Swing up -------------------------------------------- -- #
    elif env_name.startswith("CartPoleSwingUp"):
        from domain.cartpole_swingup import CartPoleSwingUpEnv

        env = CartPoleSwingUpEnv()
        if env_name.startswith("CartPoleSwingUp_Hard"):
            env.dt = 0.01
            env.t_limit = 200

    elif env_name == "SlimeVolley-Shaped-v0":
        from domain.slimevolley_shaped import SlimeVolleyShapedEnv

        env = SlimeVolleyShapedEnv(enable_curriculum=False)

    elif env_name == "SlimeVolley-Shaped-Curriculum-v0":
        from domain.slimevolley_shaped import SlimeVolleyShapedEnv

        env = SlimeVolleyShapedEnv(
            enable_curriculum=True, initial_curriculum_stage="survival"
        )

    elif env_name == "SlimeVolley-Curriculum-v0":
        # Backward compatibility alias
        from domain.slimevolley_shaped import SlimeVolleyShapedEnv

        env = SlimeVolleyShapedEnv(
            enable_curriculum=True, initial_curriculum_stage="survival"
        )

    # -- Slime Volleyball ---------------------------------------------- -- #
    elif env_name.startswith("SlimeVolley"):
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
        # Pass render_mode during initialization if needed
        render_kwargs = {}
        if render_mode:
            # Convert boolean to string if needed
            if isinstance(render_mode, bool):
                render_kwargs["render_mode"] = "human" if render_mode else None
            else:
                render_kwargs["render_mode"] = render_mode
        env = gym.make(env_name, **render_kwargs)

    if seed >= 0:
        env.seed(seed)

    logger.debug(f"Action space: {env.action_space}")
    logger.debug(f"Observation space: {env.observation_space}")

    return env
