import torch
import random
import gym
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm as _tqdm
import random
from train_QNet import *
import copy
import utils
import math


def repeat_trajectory(trajectory, seed, env_name):
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

    return environment_states, states, returns #, partial_returns


def backward_train(train, model, memory, trajectory, seed, env_name, stop_coeff, smoothing_num,
                   num_splits, max_num_episodes, batch_size, discount_factor, learn_rate,
                   get_epsilon, use_target_qnet=None, render=False):

    optimizer = optim.Adam(model.parameters(), learn_rate)

    if use_target_qnet is not None:
        target_model = copy.deepcopy(model)
    else:
        target_model, target_optim = None, None

    # Count the steps (do not reset at episode start, to compute epsilon)
    global_steps = 0
    episode_durations = []
    returns_trends = []
    disc_rewards = []
    losses = []
    trajectories = []

    splits = utils.chunks(len(trajectory), num_splits)

    environment_states, states, rewards = repeat_trajectory(
        trajectory, seed, env_name)

    for s, split in enumerate(splits):
        print("Split", s)
        # TODO - redefine getepsilon function here
        
        block_returns_trends = []
        
        for i in range(max_num_episodes):
            starting_state_idx = np.random.choice(split)
            print("\t", starting_state_idx)
            env = copy.deepcopy(environment_states[starting_state_idx])
            state = states[starting_state_idx]
            trajectory_return = rewards[starting_state_idx]
            
            duration = 0
            episode_return = 0
            disc_reward = 0
            
            current_trajectory = trajectory[:starting_state_idx]

            if use_target_qnet is not None and i % 5 == 0:
                target_model = copy.deepcopy(model)
            
            env.render() if render and i % 1 == 0 else None

            while True:
                epsilon = get_epsilon(i)
                a = select_action(model, state, epsilon)

                next_state, r, done, _ = env.step(a)
                env.render() if render and i % 1 == 0 else None

                duration += 1
                episode_return += r

                disc_reward += (discount_factor ** duration) * r

                current_trajectory.append((state, a, r, next_state, done))
                memory.push((state, a, r, next_state, done))
                loss = train(model, memory, optimizer, batch_size, discount_factor,
                             target_model=target_model)
                global_steps += 1

                if done:
                    break

                state = next_state
            
            env.close()
            
            # TODO: save it in a dictionary (for example, based on reward or duration) or do it in post process
            # saving the seed(i) is necessary for replaying the episode later
            trajectories.append((current_trajectory, seed))

            losses.append(loss)
            episode_durations.append(duration)
            returns_trends.append(episode_return)
            block_returns_trends.append(episode_return)
            
            # TODO - multiply it by gamma**len(trajectory till the starting point)
            disc_rewards.append(disc_reward)

            if len(block_returns_trends) > smoothing_num and np.mean(block_returns_trends[-smoothing_num:]) > trajectory_return * stop_coeff:
                break

    return model, episode_durations, returns_trends, disc_rewards, losses, trajectories
