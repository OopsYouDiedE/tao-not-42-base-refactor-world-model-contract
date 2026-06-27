import gymnasium as gym
import numpy as np
from craftground import make, InitialEnvironmentConfig
from stable_baselines3 import A2C

class FlattenObsWrapper(gym.Wrapper):
    """Flatten nested observation space for compatibility with SB3"""
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )

    def _flatten_obs(self, obs):
        # Extract image from nested observation if needed
        if isinstance(obs, dict):
            return obs.get('image', obs.get('rgb', np.zeros((84, 84, 3), dtype=np.uint8)))
        elif isinstance(obs, tuple):
            return obs[0] if len(obs) > 0 else np.zeros((84, 84, 3), dtype=np.uint8)
        return obs

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info

# Initialize environment with config
config = InitialEnvironmentConfig(
    image_width=84,
    image_height=84,
)
env = make(initial_env_config=config, port=8023)
env = FlattenObsWrapper(env)

# Train model
model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("a2c_craftground")

print("Training complete! Model saved as a2c_craftground")
