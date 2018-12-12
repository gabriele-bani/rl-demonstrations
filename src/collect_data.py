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

num_datapoints = 3

num_episodes = 170
discount_factor = 0.99

splits_lst = [20]
eps_lst = [0, 1]
smoothing_num = 10
stop_coeff = 5

intial_eps = 1
final_eps = 0.05


random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
env = gym.envs.make(env_name)
env.reset()

env.seed(seed)

# data = utils.load_trajectories(env_name)
data = utils.load_trajectories(env_name, filename="selected_trajectories")

data.sort_values(by="sum_reward", inplace=True)

results = []
# for index, row in data.iterrows():
for index in [0, 2, 4]:
    row = data.iloc[index]
    for eps_it in eps_lst:
        for split in splits_lst:
            results = []
            for i in range(num_datapoints):
                model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])
                testing_seed = np.random.randint(0, 5000)
                
                print(f"Starting Backward Training with eps={eps_it}, num_splits={split}, row={index}, seed={testing_seed}, {i}-th run")
                
                trajectory = row.trajectory
                seed = row.seed
                demonstration_value = row.sum_reward

                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                get_epsilon = lambda it: intial_eps - it*((intial_eps - final_eps)/eps_it)  if it < eps_it  else final_eps
                
                model, episode_durations, returns_trends, disc_rewards, losses, trajectories = backward_train(
                    train=train_QNet_true_gradient,
                    model=model,
                    memory=memory,
                    trajectory=trajectory,
                    seed=seed,
                    env_name=env_name,
                    stop_coeff=stop_coeff,
                    smoothing_num=smoothing_num,
                    num_splits=split,
                    max_num_episodes=num_episodes,
                    batch_size=batch_size,
                    discount_factor=discount_factor,
                    learn_rate=learn_rate,
                    get_epsilon=get_epsilon,
                    use_target_qnet=use_target_qnet,
                    render=render,
                    testing_seed=testing_seed,
                    verbose=False
                )
                results.append((
                    returns_trends,
                    testing_seed,
                    demonstration_value,
                    split,
                    eps_it,
                    stop_coeff,
                    smoothing_num
                ))

            utils.store_experiments(env_name, None, results)

eps_iterations = 1
intial_eps = 1
final_eps = 0.05

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
env.reset()

env.seed(seed)

results = []
for i in range(num_datapoints):
    model = QNetwork(num_inputs=num_inputs[env_name],
                    num_hidden=num_hidden, num_outputs=num_outputs[env_name])
    
    testing_seed = np.random.randint(0, 5000)
    print(f"Starting Training from scratch seed={testing_seed}, {i}-th run")
    
    get_epsilon = lambda it: 1 - it*((1 - final_eps)/eps_iterations) if it < eps_iterations else final_eps
    
    episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_QNet_true_gradient,
                                                                                  model,
                                                                                  memory,
                                                                                  env,
                                                                                  num_episodes,
                                                                                  batch_size,
                                                                                  discount_factor,
                                                                                  learn_rate,
                                                                                  get_epsilon=get_epsilon,
                                                                                  use_target_qnet=use_target_qnet,
                                                                                  seed=testing_seed,
                                                                                  render=render)
    results.append((
        rewards,
        testing_seed,
        None,
        None,
        None,
        None,
        None
    ))

utils.store_experiments(env_name, None, results)
