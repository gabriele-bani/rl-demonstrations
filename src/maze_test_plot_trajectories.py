
import random

from replay import play_episodes, play_trajectory
from backward_train_maze import backward_train_maze, train_maze

import utils
from utils import smooth
import matplotlib.pyplot as plt
import numpy as np
import pickle

env_name = "Maze_(15,15,42,1.0,1.0)"

env = utils.create_env(env_name)

seed = 34
# # whether to visualize some episodes during training
render = False

num_splits = 5
num_episodes = 300
discount_factor = 0.99

eps_iterations = 0
final_eps = 0.05

def my_smooth(x, N):
    arrays = []
    for split in range(num_splits-1):
        start = split*num_episodes
        end = (split+1)*num_episodes -1
        arrays.append(smooth(x[start:end], N))
    arrays.append(smooth(x[num_episodes*(num_splits-1):], N))
    
    return np.concatenate(arrays)


def get_epsilon(it):
    return 1 - it*((1 - final_eps)/eps_iterations) if it < eps_iterations else final_eps

def generate_plots():
    
    w = 0.5
    xmax_len = len(my_smooth(optimal, smooth_factor))
    plt.figure()
    plt.ylim((-300, -50))
    plt.hlines(-67, xmin=0, xmax=xmax_len, linewidth=0.2, color='black',
               linestyles='--', label='Optimal Trajectory')
    plt.plot(my_smooth(optimal, smooth_factor), label="Optimal Demo (G= -{})".format(67), alpha=1, linewidth=0.8, color='green')
    plt.plot(my_smooth(suboptimal, smooth_factor), label="Suboptimal Demo (G= -{})".format(139), linewidth=0.8, alpha=1, color='orange')
    plt.plot(my_smooth(bad, smooth_factor), label="Bad Demo (G= -{})".format(207), alpha=1, linewidth=0.8, color='blue')
    # plt.plot(my_smooth(returns_trends, smooth_factor), label="optimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_suboptimal, smooth_factor), label="suboptimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_bad, smooth_factor), label="bad demo", alpha=0.5)
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([optimal[i] - w*std_optimal[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([optimal[i] + w*std_optimal[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="green")
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([suboptimal[i] - w* std_suboptimal[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([suboptimal[i] + w*std_suboptimal[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="yellow")
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([bad[i] - w*std_bad[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([bad[i] + w*std_bad[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="blue")
    plt.yticks(list(plt.yticks()[0]) + [-67])
    plt.xlabel("Episode during training")
    plt.ylabel("Total Return in episode")
    plt.legend()
    plt.title('Episode Returns for each demonstration')
    # plt.show()
    plt.savefig("./plot.svg", format='svg', dpi=1000)
    return


smooth_factor = 100  # TODO   notice the smoothing factor in the plots!
(smooth_factor, optimal, suboptimal, bad, std_optimal, std_suboptimal, std_bad) = pickle.load(open('values.pickle', "rb"))

# pickle.load(open(PATHS["measures"], "rb"))
generate_plots()

# play_episodes(utils.create_env(env_name), greedy_policy, n=1, seed=trajectories[-1][1], render=True, maze=True, plotting=False)