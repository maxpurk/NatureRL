import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os
import subprocess
# from ale_py.roms import Breakout
# from ale_py import ALEInterface


# ale = ALEInterface()
# ale.loadROM(Breakout)

###################################
#Create the environment
# environment_name = "Breakout-v4"
# env = gym.make(environment_name) #
# env.close()
# env = gym.make(environment_name, render_mode ="human")
# l = env.reset()
# m = env.action_space
# i = env.observation_space
# print(f"reset: {l} actiospace: {m} observation_space: {i}")

###################################
#Explore the enviornment

# episodes = 5
# for episode in range(1,episodes+1 ):
#     obs = env.reset()
#     done = False
#     score = 0

#     while not done:
#         action = env.action_space.sample()
#         obs,reward, done,_, info = env.step(action)
#         score += reward
#     print("Episode:{} Score:{}".format(episode,score))
# env.close()


# #Initalize themodel
# log_path = os.path.join ("Training","Logs")
# model = A2C("CnnPolicy", env, verbose= 1, tensorboard_log= log_path)

# model.learn(total_timesteps=1000)

# # # # # #Save the model
# a2c_path = os.path.join("Training","Saved Models", "A2C_Breakout_Models")
# #model.save(a2c_path)

# # #load the model

# env = gym.make(environment_name, render_mode = None)
# env.reset()
# model = A2C.load("Training/Saved Models/A2C_Breakout_Models.zip", env)

# # #Evalute and Test

# results = evaluate_policy(model, env, n_eval_episodes=1, render = False)

# print(results)

#Tensorboard Command: tensorboard --logdir=Training/Logs/PPO_1

################################################################################################
#Test the model

# episodes = 5
# for episode in range(1,episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0 

#     while not done:
#         #env.render()
#         action, _  = model.predict(obs) #Now 
#         obs, reward, done_array, info = env.step(action)
#         done = done_array.any()
#         score += reward
#     print("Episode:{} Score: {}".format(episode,score))

# env.close()


################################################################################################
################################################################################################
################################################################################################


###################################
#Vectorise Environment and Train Model

# environment_name = "Breakout-v4"
# env = make_atari_env(environment_name, n_envs=4, seed = 0)
# env = VecFrameStack(env,n_stack=4)

# # env.reset()

# # #Initalize themodel
# log_path = os.path.join ("Training","Logs")
# model = A2C("CnnPolicy", env, verbose= 1, tensorboard_log= log_path)

# model.learn(total_timesteps=1000000)

# # # # # #Save the model
# a2c_path = os.path.join("Training","Saved Models", "A2C_Breakout_Models")
# model.save(a2c_path)

# # #load the model
# # model = A2C.load(a2c_path, env)

# # # #Evalute and Test

# results = evaluate_policy(model, env, n_eval_episodes=10, render=False)

# print(results)

#Tensorboard Command: tensorboard --logdir=NatureRL/Training/Logs/PPO_1

################################################################################################
#Test the model

# episodes = 5
# for episode in range(1,episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0 

#     while not done:
#         env.render()
#         action, _  = model.predict(obs) #Now 
#         obs, reward, done_array, info = env.step(action)
#         done = done_array.any()
#         score += reward
#     print("Episode:{} Score: {}".format(episode,score))

# env.close()