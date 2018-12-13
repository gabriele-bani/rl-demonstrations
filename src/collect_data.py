
import numpy as np
import torch
from train_QNet import train_QNet_true_gradient
from run_episodes import run_episodes
from memory import ReplayMemory
from QNetwork import QNetwork
import random
import gym
from backward_train import backward_train

import utils

seed_sampler = random.Random()

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

batch_size = 128
learn_rate = 1e-2
num_hidden = 128
# seed = 33
use_target_qnet = False
# whether to visualize some episodes during training
render = False

num_datapoints = 20

num_episodes = 150
discount_factor = 0.99

splits_lst = [20]
eps_lst = [1]
smoothing_num = 10
stop_coeff = 4

intial_eps = 1
final_eps = 0.05

env = gym.envs.make(env_name)

# data = utils.load_trajectories(env_name)
data = utils.load_trajectories(env_name, filename="selected_trajectories")

data.sort_values(by="sum_reward", inplace=True)

results = []
# for index, row in data.iterrows():
for i in range(num_datapoints):
    
    intial_eps = 1
    final_eps = 0.05
    learn_rate = 1e-2
    
    for index in [1, 4, 5]:
        row = data.iloc[index]
        for eps_it in eps_lst:
            for split in splits_lst:
            
                testing_seed = seed_sampler.randint(0, 5000)
                
                torch.manual_seed(testing_seed)
                model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])
                memory = ReplayMemory(2000)
                
                print(f"Starting Backward Training with eps={eps_it}, num_splits={split}, row={index}, seed={testing_seed}, {i}-th run")
                
                trajectory = row.trajectory
                seed = row.seed
                demonstration_value = row.sum_reward
                
                get_epsilon = lambda it: intial_eps - it*((intial_eps - final_eps)/eps_it) if it < eps_it else final_eps
                
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
                results = [(
                    returns_trends,
                    testing_seed,
                    demonstration_value,
                    split,
                    eps_it,
                    stop_coeff,
                    smoothing_num
                )]

                utils.store_experiments(env_name, None, results)

    
    
    testing_seed = seed_sampler.randint(0, 5000)
    
    random.seed(testing_seed)
    torch.manual_seed(testing_seed)
    np.random.seed(testing_seed)

    eps_iterations = 150
    intial_eps = 1
    final_eps = 0.05
    learn_rate = 1e-3
    get_epsilon = lambda it: 1 - it * ((1 - final_eps) / eps_iterations) if it < eps_iterations else final_eps
    
    model = QNetwork(num_inputs=num_inputs[env_name],
                    num_hidden=num_hidden, num_outputs=num_outputs[env_name])
    memory = ReplayMemory(2000)
    
    print(f"Starting Training from scratch seed={testing_seed}, {i}-th run")
    
    episode_durations, rewards, disc_rewards, losses, trajectories = run_episodes(train_QNet_true_gradient,
                                                                                  model,
                                                                                  memory,
                                                                                  env,
                                                                                  500,
                                                                                  batch_size,
                                                                                  discount_factor,
                                                                                  learn_rate,
                                                                                  get_epsilon=get_epsilon,
                                                                                  use_target_qnet=use_target_qnet,
                                                                                  seed=testing_seed,
                                                                                  render=render)
    results = [(
        rewards,
        testing_seed,
        None,
        None,
        None,
        None,
        None
    )]

    utils.store_experiments(env_name, None, results)
