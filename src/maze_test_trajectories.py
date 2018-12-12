
import random

from replay import play_episodes, play_trajectory
from backward_train_maze import backward_train_maze, train_maze

import utils
from utils import smooth
import matplotlib.pyplot as plt
import numpy as np

env_name = "Maze_(15,15,42,1.0,1.0)"

# env = gym.envs.make(env_name)
env = utils.create_env(env_name)


seed = 34
# use_target_qnet = False
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
    
    plt.figure()
    plt.ylim((-300, -50))
    plt.hlines(-67, xmin=0, xmax=len(my_smooth(returns_trends_suboptimal, smooth_factor)), linewidth=0.2, color='black',
               linestyles='--', label='optimal trajectory')
    plt.plot(my_smooth(optimal, smooth_factor), label="optimal demo", alpha=0.5)
    plt.plot(my_smooth(suboptimal, smooth_factor), label="suboptimal demo", alpha=0.5)
    plt.plot(my_smooth(bad, smooth_factor), label="bad demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends, smooth_factor), label="optimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_suboptimal, smooth_factor), label="suboptimal demo", alpha=0.5)
    # plt.plot(my_smooth(returns_trends_bad, smooth_factor), label="bad demo", alpha=0.5)
    plt.yticks(list(plt.yticks()[0]) + [-67])
    plt.legend()
    plt.title('Episode durations')
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

print("Starting Training")
for i in range(19):
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
    
bad= [x/20 for x in bad]
optimal = [x/20 for x in optimal]
suboptimal = [x/20 for x in suboptimal]

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
generate_plots()

# play_episodes(utils.create_env(env_name), greedy_policy, n=1, seed=trajectories[-1][1], render=True, maze=True, plotting=False)