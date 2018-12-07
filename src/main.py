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

import utils

dir_path = os.path.dirname(os.path.realpath(__file__))
env_name = "MountainCar-v0"
# name = "LunarLander-v2"
# name = "CartPole-v0"

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

num_episodes = 500
discount_factor = 0.99

eps_iterations = 100
final_eps = 0.05

def get_epsilon(it):
    return 1 - it*((1 - final_eps)/eps_iterations) if it < eps_iterations else final_eps


random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])

# episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_QNet, model, memory, env, num_episodes,
#                                                                 batch_size, discount_factor, learn_rate,
#                                                                 render=render)

episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_QNet_true_gradient, model, memory,
                            env, num_episodes, batch_size, discount_factor, learn_rate, get_epsilon=get_epsilon,
                            use_target_qnet=use_target_qnet, render=render)


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# only needed for cartpole, due to memory replay we might miss the losses of the first few episodes
losses[losses == None] = 0

d = {"rewards": rewards,
     "discounted rewards": disc_rewards,
     "episode durations": episode_durations,
     "loss": losses}

dir = utils.build_data_dir(env_name)

utils.store_results(env_name, d)
utils.store_model(env_name, model)
utils.store_trajectories(env_name, trajectories, None, discount_factor)

fig, axes = plt.subplots(nrows=2, ncols=2)
for i, (n, l) in enumerate(d.items()):
    print(i, n, l)
    y, x = i // 2, i % 2
    axes[y, x].plot(smooth(l, 20))
    axes[y, x].set_title(n)
plt.show()


# play some episodes to test
print("start playing episodes with the trained model")
play_episodes(env, model, 3)


# play the first 20 trajectories.
# note that to reproduce exactly the trajectory, we have to also pass the seed (trajectories[i][1])
# so that we completely match the random effects that were present in the original run
print("start replaying trajectories")
for i in range(5):
    print("replaying trajectory", -i)
    play_trajectory(env, trajectories[-i][0], seed=trajectories[-i][1])
