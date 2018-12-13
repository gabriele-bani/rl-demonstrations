import numpy as np
import torch
import random
from backward_train_maze import backward_train_maze, train_maze

import utils

seed_sampler = random.Random()


env_name = "Maze_(15,15,42,1.0,1.0)"

# whether to visualize some episodes during training
render = False

num_datapoints = 15

discount_factor = 0.99

num_episodes = 100

splits_lst = [1, 5, 8, 13, 17]

eps_lst = [0, 10, 50]

smoothing_num = 15
stop_coeff = 3

intial_eps = 1
final_eps = 0.05


env = utils.create_env(env_name)

data = utils.load_trajectories(env_name)
data_bad = utils.load_trajectories(env_name, filename='trajectories_bad')

suboptimal_trajectory = data.iloc[-1]
optimal_trajectory = data.iloc[data["sum_reward"].idxmax()]
bad_trajectory = data_bad.iloc[-1]


demonstrations = [suboptimal_trajectory, optimal_trajectory, bad_trajectory]

results = []
for i in range(num_datapoints):
    
    for idx, row in enumerate(demonstrations):
        print(idx, row.sum_reward)
        for eps_it in eps_lst:
            for split in splits_lst:
                
                testing_seed = seed_sampler.randint(0, 5000)

                print(f"Starting Training with row={idx}, eps={eps_it}, num_splits={split}, seed={testing_seed}, {i}-th run")
                
                trajectory = row.trajectory
                seed = row.seed
                demostration_value = row.sum_reward
                get_epsilon = lambda it: intial_eps - it*((intial_eps - final_eps)/eps_it) if it < eps_it else final_eps

                random.seed(seed)
                torch.manual_seed(seed)
                np.random.seed(seed)

                Q, greedy_policy, episode_durations, returns_trends, disc_rewards, trajectories = backward_train_maze(
                    trajectory=trajectory,
                    seed=seed,
                    env_name=env_name,
                    stop_coeff=stop_coeff,
                    smoothing_num=smoothing_num,
                    num_splits=split,
                    max_num_episodes=num_episodes,
                    discount_factor=discount_factor,
                    get_epsilon=get_epsilon,
                    render=render,
                    testing_seed=testing_seed,
                    verbose=False
                )
                results =[(
                    returns_trends,
                    testing_seed,
                    demostration_value,
                    split,
                    eps_it,
                    stop_coeff,
                    smoothing_num
                )]

                utils.store_experiments(env_name, env.get_params(), results)

    testing_seed = seed_sampler.randint(0, 5000)

    print(f"Starting Training from Scratch with seed={testing_seed}, {i}-th run")

    eps_iterations = 10
    final_eps = 0.05

    def get_epsilon(it):
        return 1 - it * ((1 - final_eps) / eps_iterations) if it < eps_iterations else final_eps

    Q_scratch, greedy_policy_scratch, episode_durations_scratch, returns_trends_scratch, disc_rewards_scratch, trajectories_scratch = train_maze(
        seed=testing_seed,
        env_name=env_name,
        max_num_episodes=300,
        discount_factor=discount_factor,
        get_epsilon=get_epsilon,
        render=render,
        verbose=False
    )
    results = [(
        returns_trends_scratch,
        testing_seed,
        None,
        None,
        None,
        None,
        None
    )]


    utils.store_experiments(env_name, env.get_params(), results)
