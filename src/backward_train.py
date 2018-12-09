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


def repeat_trajectory(trajectory, seed, env_name):
    # Note that step is from the end

    env = utils.create_env(env_name)

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
            raise ValueError(
                "Trajectory not copied during repeat_trajectory function!")

    total_reward = 0
    for i, reward in enumerate(rewards):
        total_reward += reward
        rewards[i] = total_reward

    rewards = [np.abs(total_reward - reward) for reward in rewards]

    return environment_states, states, rewards


def backward_train(train, model, memory, trajectory, seed, env_name, stop_coeff, smoothing_num,
                   num_splits, num_samples, max_num_episodes, batch_size, discount_factor, learn_rate,
                   get_epsilon, use_target_qnet=None, render=False):

    optimizer = optim.Adam(model.parameters(), learn_rate)

    if use_target_qnet != None:
        target_model = copy.deepcopy(model)
    else:
        target_model, target_optim = None, None

    # Count the steps (do not reset at episode start, to compute epsilon)
    global_steps = 0
    episode_durations = []
    rewards = []
    disc_rewards = []
    losses = []
    trajectories = []

    num_obs = list(range(len(trajectory)))
    splits = utils.chunks(num_obs, len(num_obs)//num_splits)
    starting_state_idxs = []

    for split in splits:
        for j in range(num_samples):
            starting_state_idxs.append(np.random.choice(split))

    environment_states, states, rewards = repeat_trajectory(
        trajectory, seed, env_name)

    for starting_state_idx in starting_state_idxs:
        env, state, reward = environment_states[starting_state_idx], states[
            starting_state_idx], rewards[starting_state_idx]
        for i in range(max_num_episodes):
            duration = 0
            reward = 0
            disc_reward = 0
            loss = -99999

            trajectory = []

            if use_target_qnet != None and i % 5 == 0:
                target_model = copy.deepcopy(model)

            env.render() if render and i % 1 == 0 else None

            while True:
                epsilon = get_epsilon(i)
                a = select_action(model, state, epsilon)

                next_state, r, done, _ = env.step(a)
                env.render() if render and i % 1 == 0 else None

                duration += 1
                reward += r

                disc_reward += (discount_factor ** duration) * r

                trajectory.append((state, a, r, next_state, done))
                memory.push((state, a, r, next_state, done))
                loss = train(model, memory, optimizer, batch_size, discount_factor,
                             target_model=target_model)
                global_steps += 1

                if done:
                    break

                state = next_state
            # TODO: save it in a dictionary (for example, based on reward or duration) or do it in post process
            # saving the seed(i) is necessary for replaying the episode later
            trajectories.append((trajectory, seed))

            losses.append(loss)
            episode_durations.append(duration)
            rewards.append(reward)
            disc_rewards.append(disc_reward)

            if np.mean(rewards[-smoothing_num:]) > reward * stop_coeff:
                break

    return episode_durations, rewards, disc_rewards, losses, trajectories
