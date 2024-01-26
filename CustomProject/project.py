import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete 
import numpy as np
import random
from typing import Any
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy

class ShowerEnv(Env):
    def __init__(self) -> None:
        super().__init__()
        #Actions we can take down, stay, up
        self.action_space =Discrete(3)
        #Temperature Array
        self.observation_space = Box(low=np.array([0], dtype=np.float32), high=np.array([100], dtype=np.float32))
        #Set start temp
        self.observation = np.array([38 + random.randint(-3, 3)], dtype=np.float32)
        # Set shower length
        self.shower_length = 60

    def step(self,action):
        # Apply action
        # 0 -1 = -1 temperature
        # 1 -1 = 0 
        # 2 -1 = 1 temperature 
        self.observation[0] += action -1
        #Reduce shower length by one second
        self.shower_length -= 1

        #Calculate reward
        if self.observation[0]>= 37 and self.observation[0] <= 39:
            reward = 1
        else: 
            reward = -1

        #check if shower is done 
        if self.shower_length <= 0:
            terminated = True
        else: 
            terminated = False

        #Apply temperature noise
        #self.state += random.randint(-1,1)
        # Set placeholder for info
        info = {}
        
        #Since Gymnasium there is also truncated (artifical constraint)
        truncated = False

        #Return step information
        return self.observation, reward, terminated, truncated, info

    def render(self):
        #Implement viz
        pass

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict]:
        
        super().reset(seed=seed, options=options)

        
        #Reset shower temperature
        self.observation[0] = 38 + random.randint(-3, 3)    
        #Reset shower time 
        self.shower_length = 60
        return self.observation, {}

env = ShowerEnv()

#Check the custom enviroment
# from stable_baselines3.common.env_checker import check_env
# check_env(env, warn=True)

# #Check the env
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         action = env.action_space.sample()
#         n_state, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
#         score+=reward
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()

# #Train the model
log_path = os.path.join("Training","Logs")
model = PPO ("MlpPolicy", env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps=200000)

# # Save the model
ppo_path = os.path.join('Training', 'Saved Models', 'PPO_ShowerModel')
model.save(ppo_path)

#load the model
# model = PPO.load(ppo_path, env = env)

#Test the model
#Number of episodes to run
num_episodes = 5

for episode in range(1, num_episodes + 1):
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0] # Must be done due to wrapped obs.

    done_array = False # must be array because of split of done into done and truncated!
    score = 0
    while not done_array:
        # Generate action from the model
        action, _states = model.predict(obs)

        # Step the environment with the generated action
        obs, reward, done, truncated, info = env.step(action)  # Adjusted to match the output

        done_array = done or truncated

        # Accumulate the reward
        score += reward

    print(f"Episode: {episode}, Score: {score}")


