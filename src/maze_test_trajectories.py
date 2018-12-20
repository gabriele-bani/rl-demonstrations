
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
    
    w = 1
    xmax_len = len(my_smooth(optimal, smooth_factor))
    plt.figure()
    plt.ylim((-300, -50))
    plt.hlines(-67, xmin=0, xmax=xmax_len, linewidth=0.2, color='black',
               linestyles='--', label='Optimal Trajectory')
    plt.plot(my_smooth(optimal, smooth_factor), label="Optimal Demo (G= -{})".format(67), alpha=1, linewidth=0.8,
             color='green')
    plt.plot(my_smooth(suboptimal, smooth_factor), label="Suboptimal Demo (G= -{})".format(139), linewidth=0.8, alpha=1,
             color='orange')
    plt.plot(my_smooth(bad, smooth_factor), label="Bad Demo (G= -{})".format(207), alpha=1, linewidth=0.8, color='blue')
    # plt.plot(my_smooth(returns_trends, smooth_factor), label="optimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_suboptimal, smooth_factor), label="suboptimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_bad, smooth_factor), label="bad demo", alpha=0.5)
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([optimal[i] - w * std_optimal[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([optimal[i] + w * std_optimal[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="green")
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([suboptimal[i] - w * std_suboptimal[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([suboptimal[i] + w * std_suboptimal[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="yellow")
    plt.fill_between(np.arange(xmax_len),
                     my_smooth([bad[i] - w * std_bad[i] for i in range(len(optimal))], smooth_factor),
                     my_smooth([bad[i] + w * std_bad[i] for i in range(len(optimal))], smooth_factor),
                     alpha=0.1, color="blue")
    plt.yticks(list(plt.yticks()[0]) + [-67])
    plt.xlabel("Episode during training")
    plt.ylabel("Total Return in episode")
    plt.legend()
    plt.title('Episode Returns for each demonstration')
    # plt.show()
    plt.savefig("./plot.svg", format='svg', dpi=1000)
    return


random.seed(seed)
# torch.manual_seed(seed)
env.seed(seed)

# model = QNetwork(num_inputs=num_inputs[env_name], num_hidden=num_hidden, num_outputs=num_outputs[env_name])


data = utils.load_trajectories(env_name)
data_bad = utils.load_trajectories(env_name, filename='trajectories_bad')

suboptimal_trajectory = data.iloc[-1]["trajectory"]
optimal_trajectory = data.iloc[data["sum_reward"].idxmax()]["trajectory"]
# bad_trajectory = data.iloc[data["sum_reward"].idxmin()]["trajectory"]
bad_trajectory = data_bad.iloc[-1]["trajectory"]
print("len bad traj", len(bad_trajectory))
seed = data.iloc[-1]["seed"]


print("Replaying training trajectory")
# play_trajectory(utils.create_env(env_name), optimal_trajectory, seed=seed, render=True)

a = []
b = []
c = []

_,_,_,optimal, disc_rewards, trajectories = backward_train_maze(
    trajectory=optimal_trajectory,
    seed=seed,
    env_name=env_name,
    stop_coeff=0.2,
    smoothing_num=5,
    num_splits=num_splits,
    # num_samples=5,
    max_num_episodes=num_episodes,
    discount_factor=discount_factor,
    get_epsilon=get_epsilon,
    render=render
)

_,_,_,suboptimal, disc_rewards_suboptimal, trajectories_suboptimal = backward_train_maze(
    trajectory=suboptimal_trajectory,
    seed=seed,
    env_name=env_name,
    stop_coeff=0.2,
    smoothing_num=5,
    num_splits=num_splits,
    # num_samples=5,
    max_num_episodes=num_episodes,
    discount_factor=discount_factor,
    get_epsilon=get_epsilon,
    render=render
)

_,_,_,bad, disc_rewards_bad, trajectories_bad = backward_train_maze(
    trajectory=bad_trajectory,
    seed=seed,
    env_name=env_name,
    stop_coeff=0.2,
    smoothing_num=5,
    num_splits=num_splits,
    # num_samples=5,
    max_num_episodes=num_episodes,
    discount_factor=discount_factor,
    get_epsilon=get_epsilon,
    render=render
)

a.append(optimal)
b.append(suboptimal)
c.append(bad)
EXP = 100

print("Starting Training")
for i in range(EXP-1):
    Q, greedy_policy, episode_durations, returns_trends, disc_rewards, trajectories = backward_train_maze(
                                                                                           trajectory=optimal_trajectory,
                                                                                           seed=seed,
                                                                                           env_name=env_name,
                                                                                           stop_coeff=0.2,
                                                                                           smoothing_num=5,
                                                                                           num_splits=num_splits,
                                                                                           # num_samples=5,
                                                                                           max_num_episodes=num_episodes,
                                                                                           discount_factor=discount_factor,
                                                                                           get_epsilon=get_epsilon,
                                                                                           render=render
                                                                                    )
    
    Q_suboptimal, greedy_policy_suboptimal, episode_durations_suboptimal, returns_trends_suboptimal, disc_rewards_suboptimal, trajectories_suboptimal = backward_train_maze(
                                                                                           trajectory=suboptimal_trajectory,
                                                                                           seed=seed,
                                                                                           env_name=env_name,
                                                                                           stop_coeff=0.2,
                                                                                           smoothing_num=5,
                                                                                           num_splits=num_splits,
                                                                                           # num_samples=5,
                                                                                           max_num_episodes=num_episodes,
                                                                                           discount_factor=discount_factor,
                                                                                           get_epsilon=get_epsilon,
                                                                                           render=render
                                                                                    )
    
    Q_bad, greedy_policy_bad, episode_durations_bad, returns_trends_bad, disc_rewards_bad, trajectories_bad = backward_train_maze(
                                                                                           trajectory=bad_trajectory,
                                                                                           seed=seed,
                                                                                           env_name=env_name,
                                                                                           stop_coeff=0.2,
                                                                                           smoothing_num=5,
                                                                                           num_splits=num_splits,
                                                                                           # num_samples=5,
                                                                                           max_num_episodes=num_episodes,
                                                                                           discount_factor=discount_factor,
                                                                                           get_epsilon=get_epsilon,
                                                                                           render=render
                                                                                    )
    bad = [sum(x) for x in zip(returns_trends_bad, bad)]
    optimal = [sum(x) for x in zip(returns_trends, optimal)]
    suboptimal= [sum(x) for x in zip(returns_trends_suboptimal, suboptimal)]
    
    a.append(returns_trends)
    b.append(returns_trends_suboptimal)
    c.append(returns_trends_bad)
    
bad = [x/EXP for x in bad]
optimal = [x/EXP for x in optimal]
suboptimal = [x/EXP for x in suboptimal]

# std_optimal = [10 for x in optimal]
# std_suboptimal = [10 for x in optimal]
# std_bad = [10 for x in optimal]

std_optimal = np.std(np.array(a), axis=0)
std_suboptimal = np.std(np.array(b), axis=0)
std_bad = np.std(np.array(c), axis=0)

# Q_scratch, greedy_policy_scratch, episode_durations_scratch, returns_trends_scratch, disc_rewards_scratch, trajectories_scratch = train_maze(
#                                                                                        seed=seed,
#                                                                                        env_name=env_name,
#                                                                                        # num_samples=5,
#                                                                                        max_num_episodes=600,
#                                                                                        discount_factor=discount_factor,
#                                                                                        get_epsilon=get_epsilon,
#                                                                                        render=render
#                                                                                 )



# print("Repeating the last training episode")
# play_trajectory(utils.create_env(env_name), trajectories[-1][0], seed=trajectories[-1][1], render=True)

print("\nTesting the final greedy policy:")
print("Optimal converged in ", len(episode_durations), "episodes")
play_episodes(utils.create_env(env_name), greedy_policy, n=1, seed=trajectories[-1][1], render=False, maze=True, plotting=False)
print("Suboptimal converged in ", len(episode_durations_suboptimal), "episodes")
play_episodes(utils.create_env(env_name), greedy_policy_suboptimal, n=1, seed=trajectories[-1][1], render=False, maze=True, plotting=False)
print("Bad converged in ", len(episode_durations_bad), "episodes")
play_episodes(utils.create_env(env_name), greedy_policy_bad, n=1, seed=trajectories[-1][1], render=False, maze=True, plotting=False)
# print("Scratch converged in ", lenepisode_durations_scratch), "episodes"
# play_episodes(utils.create_env(env_name), greedy_policy_scratch, n=1, seed=trajectories[-1][1], render=False, maze=True, plotting=False)

smooth_factor = 100  # TODO   notice the smoothing factor in the plots!
pickle.dump((smooth_factor, optimal, suboptimal, bad, std_optimal, std_suboptimal, std_bad), open('values.pickle', "wb"))

# pickle.load(open(PATHS["measures"], "rb"))
generate_plots()

# play_episodes(utils.create_env(env_name), greedy_policy, n=1, seed=trajectories[-1][1], render=True, maze=True, plotting=False)