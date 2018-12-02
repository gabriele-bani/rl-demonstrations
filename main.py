import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm as _tqdm
from train_functions import train, train_true_gradient
from run_episodes import run_episodes
from memory import ReplayMemory
from QNetwork import QNetwork
import random
import gym
from replay import play_episodes, play_trajectory

name = "MountainCar-v0"
# name = "LunarLander-v2"
# name = "CartPole-v0"

env = gym.envs.make(name)

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

num_episodes = 50
batch_size = 64
discount_factor = 1
learn_rate = 1e-3
memory = ReplayMemory(10000)
num_hidden = 128
seed = 42

# whether to visualize some episodes during training
render = True


random.seed(seed)
torch.manual_seed(seed)
env.seed(seed)

model = QNetwork(num_inputs=num_inputs[name], num_hidden=num_hidden, num_outputs=num_outputs[name])

episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train, model, memory, env, num_episodes,
                                                                batch_size, discount_factor, learn_rate,
                                                                render=render)
#
# episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_true_gradient, model, memory, env, num_episodes,
#                                                                 batch_size, discount_factor, learn_rate, render=render)


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


fig, axes = plt.subplots(nrows=3, ncols=2)
coordinates = [(0, 0), (0, 1), (1, 0), (1, 1)]
values = [rewards, disc_rewards, episode_durations, losses]
# only needed for cartpole, due to memory replay we might miss the losses of the first few episodes
losses[losses == None] = 0

names = ["rewards", "discounted rewards", "episode durations", "loss"]
for i, (c, l) in enumerate(zip(coordinates, values)):
    print(i, names[i], c, l)
    axes[c].plot(smooth(l, 20))
    axes[c].set_title(names[i])
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

