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
from backward_train import backward_train, repeat_trajectory
from PolynomialNetwork import PolynomialNetwork

import utils

env_name = "MountainCar-v0"


num_inputs = {
    "MountainCar-v0": 2,
    "LunarLander-v2": 8,
    "CartPole-v0": 4,
}

num_outputs = {
    "MountainCar-v0": 3,
    "LunarLander-v2": 4,
    "CartPole-v0": 2,
}

batch_size = 64
learn_rate = 1e-2
memory = ReplayMemory(2000)
num_hidden = 128
seed = 33
use_target_qnet = False
# whether to visualize some episodes during training
render = False

num_episodes = 150
discount_factor = 0.99

num_splits = 20
smoothing_num = 20
stop_coeff = 5

eps_iterations = 1
intial_eps = 0.5
final_eps = 0.05

# alpha = np.power(final_eps/intial_eps, 1/eps_iterations)

def get_epsilon(it):
    return intial_eps - it*((intial_eps - final_eps)/eps_iterations) if it < eps_iterations else final_eps
    # return intial_eps * alpha**it if it < eps_iterations else final_eps


random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.envs.make(env_name)

env.seed(seed)

model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])
# model = PolynomialNetwork(num_outputs=num_outputs[env_name], poly_order=2)


# data = utils.load_trajectories(env_name)
data = utils.load_trajectories(env_name, filename="selected_trajectories")

# data.sort_values(by="sum_reward", inplace=True)

print(data.sum_reward)

# row = data["sum_reward"].idxmax()
row = 1

print("Best Trajectory: {} with return {}".format(row, data.iloc[row].sum_reward))

trajectory = data.iloc[row]["trajectory"]
seed = data.iloc[row]["seed"]


# print("Replaying training trajectory")
# play_trajectory(env, trajectory, seed=seed, render=True)


print("Starting Training")
model, episode_durations, returns_trends, disc_rewards, losses, trajectories = backward_train(
                                                                                       train=train_QNet_true_gradient,
                                                                                       model=model,
                                                                                       memory=memory,
                                                                                       trajectory=trajectory,
                                                                                       seed=seed,
                                                                                       env_name=env_name,
                                                                                       stop_coeff=stop_coeff,
                                                                                       smoothing_num=smoothing_num,
                                                                                       num_splits=num_splits,
                                                                                       # num_samples=5,
                                                                                       max_num_episodes=num_episodes,
                                                                                       batch_size=batch_size,
                                                                                       discount_factor=discount_factor,
                                                                                       learn_rate=learn_rate,
                                                                                       get_epsilon=get_epsilon,
                                                                                       use_target_qnet=use_target_qnet,
                                                                                       render=render
                                                                                )

print("Trained in", len(episode_durations), " episodes")

print("Repeating the last training episode")
play_trajectory(env, trajectories[-1][0], seed=trajectories[-1][1], render=True)


print("Testing the model")
play_episodes(env, model, n=5, seed=trajectories[-1][1], render=True)

print("Fine-training the model")
eps_iterations = 10
intial_eps = 0.1
final_eps = 0.01

def get_epsilon(it):
    return intial_eps - it*((intial_eps - final_eps)/eps_iterations) if it < eps_iterations else final_eps

episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_QNet_true_gradient,
                                                                              model,
                                                                              memory,
                                                                              env,
                                                                              20,
                                                                              batch_size,
                                                                              discount_factor,
                                                                              learn_rate,
                                                                              get_epsilon=get_epsilon,
                                                                              use_target_qnet=use_target_qnet,
                                                                              render=render)

print("Repeating the last training episode")
play_trajectory(env, trajectories[-1][0], seed=trajectories[-1][1], render=True)


print("Testing the model")
play_episodes(env, model, n=5, seed=trajectories[-1][1], render=True)
