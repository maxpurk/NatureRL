#Import Dependencies
import gym
import os
import tensorboard
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#Load Environment
environment_name = "CartPole-v1"
env = gym.make(environment_name) # render_mode = "Human"
################################################################################################
# #1. Understand the environment
# episodes = 1
# for episode in range(1,episodes+1):
#     state = env.reset()
#     done = False
#     score = 0 

#     while not done:
#         #env.render()
#         action = env.action_space.sample()
#         n_state, reward, done, _, info = env.step(action)
#         score += reward
#     print("Episode:{} Score: {}".format(episode,score))

# env.close()

################################################################################################
# 2. Train a model
log_path = os.path.join("Training","Logs")

env = DummyVecEnv([lambda: env])

# model = PPO("MlpPolicy", env, verbose= 0, tensorboard_log= log_path)

# model.learn(total_timesteps=20000)
################################################################################################
# 3. Save the model
# PPO_Path = os.path.join("Training", "Saved Models", "PPO_Model_Cartpole" )
# model.save(PPO_Path)


################################################################################################
#4. Evaluation

# t = evaluate_policy(model, env, n_eval_episodes=2, render = True)
# print(type(t))

#env.close()

################################################################################################

#5. Test model
# episodes = 5
# for episode in range(1,episodes+1):
#     obs = env.reset()
#     done = False
#     score = 0 

#     while not done:
#         #env.render()
#         action, _  = model.predict(obs) #Now 
#         obs, reward, done, info = env.step(action)
#         score += reward
#     print("Episode:{} Score: {}".format(episode,score))

# env.close()
################################################################################################
# 6. Viewing Logs in Tensorboard

#Tensorboard Command: tensorboard --logdir=/Users/max/dev/NatureRL/Training/Logs/PPO_1

################################################################################################

#7. Applying Callbacks
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

# save_path = os.path.join("Training","Saved Models")
# stop_callback = StopTrainingOnRewardThreshold(reward_threshold= 200, verbose= 1)
# eval_callback = EvalCallback(env,
#                              callback_on_new_best= stop_callback,
#                              eval_freq= 10000,
#                              best_model_save_path = save_path ,
#                              verbose =1)

# # model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log= log_path)

# # model.learn(total_timesteps= 10000, callback= eval_callback)

# #Changing Policies
# net_arch = dict(pi=[128,128,128,128], vf=[128,128,128,128])

# model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_path, policy_kwargs= {"net_arch":net_arch})
# model.learn(total_timesteps=10000, callback=eval_callback)

#Using an Alternate Algorithm
from stable_baselines3 import DQN

model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10000)


