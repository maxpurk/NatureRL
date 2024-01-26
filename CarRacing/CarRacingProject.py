import os 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np

#Create the environment
# environment_name = "CarRacing-v2"
# env = gym.make(environment_name, render_mode = None) # human

# # ################################################################################################

## Visualize the environment, random actions, printing action space and observation space
# episodes = 5
# for episode in range(1, episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 
    
#     while not done:
#         #env.render()
#         action = env.action_space.sample()
#         n_state, reward, terminated, truncated , info = env.step(action)
#         score+=reward
#         done = terminated or truncated
#     print('Episode:{} Score:{}'.format(episode, score))
# env.close()
# print(env.action_space)
# print(env.observation_space)

################################################################################################
#Train the model
# log_path = os.path.join("Training", "Logs")
# model = PPO ("CnnPolicy", env, verbose = 1, tensorboard_log= log_path, device = "mps")
# model.learn(total_timesteps =500000)
# ppo_path = os.path.join('Training', 'Saved Models', 'PPO_Driving_model')
# model.save(ppo_path)

################################################################################################
####Load and test the model

environment_name = "CarRacing-v2"
env = gym.make(environment_name, render_mode='human')
#Load the trained model
model_path = "Training/Saved Models/PPO_Driving_model.zip"
model = PPO.load(model_path, env=env)

# Number of episodes to run
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

# Close the environment
env.close()



