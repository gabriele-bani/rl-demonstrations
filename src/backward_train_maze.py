import copy
import utils
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm as _tqdm
import random

from collections import defaultdict
import numpy as np


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """
    
    def policy_fn(observation):
        q = Q[observation]
        
        greedy = np.zeros(nA)
        greedy[np.argmax(q)] = 1
        
        random = np.ones(nA) / nA
        
        return (1 - epsilon) * greedy + epsilon * random
    
    return policy_fn


def make_greedy_policy(Q):
    
    def policy_fn(observation):
        return np.argmax(Q[observation])
    
    return policy_fn


def repeat_trajectory_maze(trajectory, seed, env_name):
    # Note that step is from the end

    env = utils.create_env(env_name)
    
    assert seed == int(seed)
    seed = int(seed)
    
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)

    env.reset()

    environment_states = []
    states = []
    rewards = []

    for observation in trajectory:
        environment_states.append(copy.deepcopy(env))
        obs_s, obs_a, obs_r, obs_ns, obs_done = observation
        states.append(obs_s)
        rewards.append(obs_r)
        next_state, r, done, _ = env.step(obs_a)
        if np.any(obs_ns != next_state) or r != obs_r or done != obs_done:
            raise ValueError("Trajectory not copied during repeat_trajectory function!")

    # total_reward = 0
    # for i, reward in enumerate(rewards):
    #     total_reward += reward
    #     rewards[i] = total_reward
    # rewards = [np.abs(total_reward - reward) for reward in rewards]
    
    rewards = np.array(rewards)
    returns = np.cumsum(rewards[::-1])[::-1]
    # partial_returns = np.cumsum(rewards)

    return environment_states, states, returns, env.action_space.n #, partial_returns


def backward_train_maze(trajectory, seed, env_name, stop_coeff, smoothing_num,
                        num_splits, max_num_episodes, discount_factor,
                        get_epsilon, render=False, Q=None):
    
    # Count the steps (do not reset at episode start, to compute epsilon)
    global_steps = 0
    alpha = 0.5  # TODO is 0.5 ok?
    episode_durations = []
    returns_trends = []
    disc_rewards = []
    trajectories = []
    last_split_steps = 0
    
    splits = utils.chunks(len(trajectory), num_splits)
    
    environment_states, states, real_returns, n_actions = repeat_trajectory_maze(trajectory, seed, env_name)
    
    for s, split in enumerate(splits):
        print("Split", s)
        
        victories = []

        epsilon = get_epsilon(0)
        
        if Q is None:
            Q = defaultdict(lambda: np.zeros(env.action_space.n))

        # The policy we're following
        policy = make_epsilon_greedy_policy(Q, epsilon, n_actions)
        
        for i in range(max_num_episodes):
            
            starting_state_idx = np.random.choice(split)
            print("\t{}".format(starting_state_idx))
            env = copy.deepcopy(environment_states[starting_state_idx])
            state = states[starting_state_idx]
            
            duration = 0
            episode_return = 0
            disc_reward = 0
            last_split_steps = i
            
            current_trajectory = trajectory[:starting_state_idx]
            
            env.render() if render and i % 1 == 0 else None
            
            while True:
                epsilon = get_epsilon(i)
                a = np.argmax(np.random.multinomial(1, policy(state)))
                
                next_state, r, done, _ = env.step(a)
                env.render() if render and i % 1 == 0 else None
                
                if done:
                    break

                Q[state][a] += alpha * (r + discount_factor *
                                              max(Q[next_state]) - Q[state][a])
                policy = make_epsilon_greedy_policy(Q, epsilon, n_actions)
                
                duration += 1
                episode_return += r
                disc_reward += (discount_factor ** duration) * r
                global_steps += 1
                
                current_trajectory.append((state, a, r, next_state, done))
                
                state = next_state
            
            env.close()
            
            print("\t\teps = {}; return = {}; expected return = {}".format(epsilon, episode_return,
                                                                           real_returns[starting_state_idx]))
            
            # TODO: save it in a dictionary (for example, based on reward or duration) or do it in post process
            # saving the seed(i) is necessary for replaying the episode later
            trajectories.append((current_trajectory, seed))
            
            # losses.append(loss)
            episode_durations.append(duration + starting_state_idx)
            starting_return = real_returns[0] - real_returns[starting_state_idx]
            returns_trends.append(episode_return + starting_return)
            dr = episode_return - real_returns[starting_state_idx]
            victory = dr >= -0.2 * abs(real_returns[starting_state_idx])
            # victory = int(episode_return > real_returns[starting_state_idx])
            # TODO look at the plots... suboptimal and bad do better than the baseline easily, so they go to the
            # TODO next split even if the learned policy is still very bad!
            victories.append(victory)
            
            # TODO - multiply it by gamma**len(trajectory till the starting point)
            disc_rewards.append(disc_reward)
            num_recent_victories = np.sum(victories[-smoothing_num:])
            print("\t\tNumber of Recent Victories ", num_recent_victories)

            # if len(victories) > smoothing_num and num_recent_victories >= stop_coeff:
            if len(victories) > smoothing_num and num_recent_victories >= stop_coeff:
                # if num_recent_victories >= stop_coeff:
                break
        
        print("Split", s, "finished in", i + 1, "episodes out of ", max_num_episodes)

    # Added

    eps_iterations = 20
    final_eps = 0.05
    initial_eps = 0.8

    def get_epsilon(it):
        return initial_eps - it*((initial_eps - final_eps)/eps_iterations) if it < eps_iterations else final_eps
    
    Q, greedy_policy, episode_durations_final, returns_trends_final, disc_rewards_final, trajectories_final = train_maze(
                                                                                           seed=seed,
                                                                                           env_name=env_name,
                                                                                           # num_samples=5,
                                                                                           max_num_episodes=100,
                                                                                           discount_factor=discount_factor,
                                                                                           get_epsilon=get_epsilon,
                                                                                           render=render,
                                                                                            Q=Q,
                                                                                            begin_at_step=last_split_steps
                                                                                    )

        
    greedy_policy = make_greedy_policy(Q)
    
    return Q, greedy_policy, episode_durations+episode_durations_final, returns_trends+returns_trends_final, disc_rewards+disc_rewards_final, trajectories+trajectories_final


def train_maze(seed, env_name, max_num_episodes, discount_factor,
                        get_epsilon, render=False, Q=None, begin_at_step=0):
    
    # Count the steps (do not reset at episode start, to compute epsilon)
    global_steps = 0
    alpha = 0.5  # TODO is 0.5 ok?
    episode_durations = []
    returns_trends = []
    disc_rewards = []
    trajectories = []
    
    epsilon = get_epsilon(0)

    env = utils.create_env(env_name)
    n_actions = env.action_space.n

    assert seed == int(seed)
    seed = int(seed)
    random.seed(seed)
    # env.seed(seed)  # results in: WARN: Could not seed environment <MazeEnv instance>
    
    if Q is None:
        Q = defaultdict(lambda: np.zeros(n_actions))
    
    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, n_actions)
    
    for i in range(begin_at_step, begin_at_step + max_num_episodes):
    
        state = env.reset()
        
        duration = 0
        episode_return = 0
        disc_reward = 0
        current_trajectory = []
        
        env.render() if render and i % 1 == 0 else None
        
        while True:
            epsilon = get_epsilon(i)
            a = np.argmax(np.random.multinomial(1, policy(state)))
            
            next_state, r, done, _ = env.step(a)
            env.render() if render and i % 1 == 0 else None
            
            if done:
                break
            
            Q[state][a] += alpha * (r + discount_factor *
                                    max(Q[next_state]) - Q[state][a])
            policy = make_epsilon_greedy_policy(Q, epsilon, n_actions)
            
            duration += 1
            episode_return += r
            disc_reward += (discount_factor ** duration) * r
            global_steps += 1
            current_trajectory.append((state, a, r, next_state, done))
            
            state = next_state
        
        env.close()
        
        print("\t\teps = {}; return = {}".format(epsilon, episode_return))
        
        # saving the seed(i) is necessary for replaying the episode later
        trajectories.append((current_trajectory, seed))
        
        episode_durations.append(duration)
        returns_trends.append(episode_return)
        disc_rewards.append(disc_reward)
        
    greedy_policy = make_greedy_policy(Q)
    
    return Q, greedy_policy, episode_durations, returns_trends, disc_rewards, trajectories
