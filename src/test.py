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

import utils

env_name = "MountainCar-v0"

env = gym.envs.make(env_name)

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
learn_rate = 1e-3
memory = ReplayMemory(2000)
num_hidden = 128
seed = 34
use_target_qnet = False
# whether to visualize some episodes during training
render = False

num_episodes = 250
discount_factor = 0.99

eps_iterations = 100
final_eps = 0.05

def get_epsilon(it):
    return 1 - it*((1 - final_eps)/eps_iterations) if it < eps_iterations else final_eps


random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])


data = utils.load_trajectories(env_name)

trajectory = data.iloc[-1]["trajectory"]
seed = data.iloc[-1]["seed"]



# for l in utils.chunks(100, 17):
#     print(l)

# assert False

model, episode_durations, returns_trends, disc_rewards, losses, trajectories = backward_train(
                                                                                       train=train_QNet_true_gradient,
                                                                                       model=model,
                                                                                       memory=memory,
                                                                                       trajectory=trajectory,
                                                                                       seed=seed,
                                                                                       env_name=env_name,
                                                                                       stop_coeff=0.9,
                                                                                       smoothing_num=5,
                                                                                       num_splits=5,
                                                                                       # num_samples=5,
                                                                                       max_num_episodes=num_episodes,
                                                                                       batch_size=batch_size,
                                                                                       discount_factor=discount_factor,
                                                                                       learn_rate=learn_rate,
                                                                                       get_epsilon=get_epsilon,
                                                                                       use_target_qnet=None,
                                                                                       render=render
                                                                                )


play_trajectory(utils.create_env(env_name), trajectories[-1][0], seed=trajectories[-1][1], render=True)


play_episodes(utils.create_env(env_name), model, n=5, seed=trajectories[-1][1], render=True)
