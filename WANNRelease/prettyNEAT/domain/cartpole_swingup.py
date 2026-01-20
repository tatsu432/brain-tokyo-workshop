"""
Cart pole swing-up: Original version from:
https://github.com/zuoxingdong/DeepPILCO/blob/master/cartpole_swingup.py

Modified so that done=True when x is outside of -2.4 to 2.4
Reward is also reshaped to be similar to PyBullet/roboschool version

More difficult, since dt is 0.05 (not 0.01), and only 200 timesteps
"""

import logging
import math
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils.seeding import np_random
import numpy as np

logger = logging.getLogger(__name__)

# Try to import pygame for rendering
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not available. Rendering will be disabled.")

class CartPoleSwingUpEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l_base = 0.6 # base pole length
        self.l = self.l_base # simulated pole length (see setEnv below)
        self.m_p_l = (self.m_p*self.l)
        self.force_mag = 10.0
        #self.dt = 0.05  # slower reaction (hard mode)
        self.dt = 0.01  # faster reaction (easy mode)
        self.b = 0.1  # friction coefficient

        self.t = 0 # timestep
        #self.t_limit = 200
        self.t_limit = 1000

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max], dtype=np.float32)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.noise = 0
        self._render_warning_shown = False

    def setEnv(self, envChange):
        '''
        Changes the environment, envChange is the percent change of parameter
        '''
        self.l = self.l_base*envChange

    def setNoise(self, noiseVariance):
        '''
        Changes the leven of input noise
        '''
        self.noise = noiseVariance

    def seed(self, seed=None):
        self.np_random, seed = np_random(seed)
        return [seed]

    def stateUpdate(self,action,state, noise=0):
        x, x_dot, theta, theta_dot = state
        x     += np.random.randn() * noise
        theta += np.random.randn() * noise

        s = math.sin(theta)
        c = math.cos(theta)

        xdot_update = (-2*self.m_p_l*(theta_dot**2)*s + 3*self.m_p*self.g*s*c + 4*action - 4*self.b*x_dot)/(4*self.total_m - 3*self.m_p*c**2)

        thetadot_update = (-3*self.m_p_l*(theta_dot**2)*s*c + 6*self.total_m*self.g*s + 6*(action - self.b*x_dot)*c)/(4*self.l*self.total_m - 3*self.m_p_l*c**2)

        x = x + x_dot*self.dt
        theta = theta + theta_dot*self.dt

        x_dot = x_dot + xdot_update*self.dt
        theta_dot = theta_dot + thetadot_update*self.dt  

        return (x, x_dot, theta, theta_dot)

    def step(self, action):
        # Valid action
        action = np.clip(action, -1.0, 1.0)[0]
        action *= self.force_mag

        noise_obs = self.stateUpdate(action, self.state, noise=self.noise)
        self.state = self.stateUpdate(action, self.state)

        x,x_dot,theta,theta_dot = self.state


        done = False
        if  x < -self.x_threshold or x > self.x_threshold:
          done = True

        self.t += 1
        if self.t >= self.t_limit:
          done = True

        # Reward staying in the middle
        reward_theta = (np.cos(theta)+1.0)/2.0
        reward_x = np.cos((x/self.x_threshold)*(np.pi/2.0))

        reward = reward_theta*reward_x
        #reward = (np.cos(theta)+1.0)/2.0

        x,x_dot,theta,theta_dot = noise_obs
        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])

        return obs, reward, done, {}

    def reset(self):
        #self.state = self.np_random.normal(loc=np.array([0.0, 0.0, 30*(2*np.pi)/360, 0.0]), scale=np.array([0.0, 0.0, 0.0, 0.0]))
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        self.steps_beyond_done = None
        self.t = 0 # timestep
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x,x_dot,np.cos(theta),np.sin(theta),theta_dot])
        return obs

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                if PYGAME_AVAILABLE:
                    pygame.quit()
                self.viewer = None
            return

        if not PYGAME_AVAILABLE:
            return None

        screen_width = 600
        screen_height = 600

        world_width = 5  # max visible position of cart
        scale = screen_width/world_width
        carty = screen_height/2 # TOP OF CART
        polewidth = 6.0
        polelen = scale*self.l  # 0.6 or self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            try:
                pygame.init()
                self.viewer = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("CartPole Swing-Up")
                self.clock = pygame.time.Clock()
            except pygame.error as e:
                if not self._render_warning_shown:
                    print(f"Warning: Could not initialize pygame display: {e}")
                    print("Running in headless mode. Rendering disabled.")
                    self._render_warning_shown = True
                return None

        if self.state is None:
            return None

        # Handle pygame events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None

        # Fill background
        self.viewer.fill((255, 255, 255))

        # Get cart position
        x_pos, _, theta, _ = self.state
        cartx = int(x_pos * scale + screen_width / 2.0)
        carty_int = int(carty)

        # Draw track
        track_y = carty_int + int(cartheight/2) + int(cartheight/4)
        track_left = int(screen_width/2 - self.x_threshold * scale)
        track_right = int(screen_width/2 + self.x_threshold * scale)
        pygame.draw.line(self.viewer, (0, 0, 0), (track_left, track_y), (track_right, track_y), 2)

        # Draw cart
        cart_rect = pygame.Rect(
            cartx - int(cartwidth/2),
            carty_int - int(cartheight/2),
            int(cartwidth),
            int(cartheight)
        )
        pygame.draw.rect(self.viewer, (255, 0, 0), cart_rect)

        # Draw wheels
        wheel_radius = int(cartheight/4)
        pygame.draw.circle(self.viewer, (0, 0, 0),
                          (cartx - int(cartwidth/2), carty_int + int(cartheight/2)),
                          wheel_radius)
        pygame.draw.circle(self.viewer, (0, 0, 0),
                          (cartx + int(cartwidth/2), carty_int + int(cartheight/2)),
                          wheel_radius)

        # Draw pole
        pole_end_x = cartx + int(polelen * np.sin(theta))
        pole_end_y = carty_int - int(polelen * np.cos(theta))
        pygame.draw.line(self.viewer, (0, 0, 255), (cartx, carty_int),
                        (pole_end_x, pole_end_y), int(polewidth))

        # Draw axle
        pygame.draw.circle(self.viewer, (25, 255, 255), (cartx, carty_int), int(polewidth/2))

        # Draw pole bob
        pygame.draw.circle(self.viewer, (0, 0, 0), (pole_end_x, pole_end_y), int(polewidth/2))

        pygame.display.flip()
        self.clock.tick(50)  # 50 FPS

        if mode == 'rgb_array':
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.viewer)), axes=(1, 0, 2)
            )

        return None
