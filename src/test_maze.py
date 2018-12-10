import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
from train_QNet import train_QNet, train_QNet_true_gradient
from run_episodes import run_episodes
from memory import ReplayMemory
from QNetwork import QNetwork
import random
import gym
from replay import play_episodes, play_trajectory
from backward_train_maze import backward_train_maze, train_maze

import utils

env_name = "Maze_(15,15,42,1.0,1.0)"

# env = gym.envs.make(env_name)
env = utils.create_env(env_name)

# num_inputs = {
#     "MountainCar-v0": 2,
#     "LunarLander-v2": 8,
#     "CartPole-v0": 4,
# }
#
# num_outputs = {
#     "MountainCar-v0": 3,
#     "LunarLander-v2": 4,
#     "CartPole-v0": 2,
# }
#
# batch_size = 64
# learn_rate = 1e-3
# memory = ReplayMemory(2000)
# num_hidden = 128
seed = 34
# use_target_qnet = False
# # whether to visualize some episodes during training
render = False

num_episodes = 100
discount_factor = 0.99

eps_iterations = 100
final_eps = 0.05

def get_epsilon(it):
    return 1 - it*((1 - final_eps)/eps_iterations) if it < eps_iterations else final_eps


random.seed(seed)
# torch.manual_seed(seed)
env.seed(seed)

# model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])


data = utils.load_trajectories(env_name)

trajectory = data.iloc[-1]["trajectory"]
# trajectory = data.iloc[data["sum_reward"].idxmax()]["trajectory"]
seed = data.iloc[-1]["seed"]


print("Replaying training trajectory")
# play_trajectory(utils.create_env(env_name), trajectory, seed=seed, render=True)


print("Starting Training")
Q, greedy_policy, episode_durations, returns_trends, disc_rewards, trajectories = backward_train_maze(
                                                                                       trajectory=trajectory,
                                                                                       seed=seed,
                                                                                       env_name=env_name,
                                                                                       stop_coeff=0.2,
                                                                                       smoothing_num=5,
                                                                                       num_splits=10,
                                                                                       # num_samples=5,
                                                                                       max_num_episodes=num_episodes,
                                                                                       discount_factor=discount_factor,
                                                                                       get_epsilon=get_epsilon,
                                                                                       render=render
                                                                                )

# Q, greedy_policy, episode_durations, returns_trends, disc_rewards, trajectories = train_maze(
#                                                                                        seed=seed,
#                                                                                        env_name=env_name,
#                                                                                        # num_samples=5,
#                                                                                        max_num_episodes=num_episodes,
#                                                                                        discount_factor=discount_factor,
#                                                                                        get_epsilon=get_epsilon,
#                                                                                        render=render
#                                                                                 )


print("Trained in", len(episode_durations), " episodes")

print("Repeating the last training episode")
play_trajectory(utils.create_env(env_name), trajectories[-1][0], seed=trajectories[-1][1], render=True)


print("Testing the model")
play_episodes(utils.create_env(env_name), greedy_policy, n=1, seed=trajectories[-1][1], render=True, maze=True)
