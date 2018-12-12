from train_QNet import *
import torch
import numpy as np
from torch import optim
import random
import copy
import utils
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm as _tqdm
import random


def repeat_trajectory(trajectory, seed, env_name):
    # Note that step is from the end

    env = utils.create_env(env_name)
    
    assert seed == int(seed)
    seed = int(seed)

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
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
        if np.any(obs_ns - next_state > 1e-12) or r != obs_r or done != obs_done:
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
                   get_epsilon, use_target_qnet=None, render=False, testing_seed=None):

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

    environment_states, states, real_returns = repeat_trajectory(trajectory, seed, env_name)
    
    if testing_seed is not None:
        seed = testing_seed

    for s, split in enumerate(splits):
        print("Split", s)
        
        victories = []
        
        for i in range(max_num_episodes):
            
            starting_state_idx = np.random.choice(split)
            # print("\t{}".format(starting_state_idx))
            env = copy.deepcopy(environment_states[starting_state_idx])
            env.seed(int(seed + 1000 * s + 7 * i))
            state = states[starting_state_idx]
            
            duration = 0
            episode_return = 0
            disc_reward = 0
            
            current_trajectory = trajectory[:starting_state_idx]

            if use_target_qnet is not None and i % 5 == 0:
                target_model = copy.deepcopy(model)
            
            env.render() if render and i % 1 == 0 else None

            while True:
                # if duration < len(split):
                #     epsilon = get_epsilon(i)
                # else:
                #     epsilon = get_epsilon(10000)

                epsilon = get_epsilon(i)
                a = select_action(model, state, epsilon)

                next_state, r, done, _ = env.step(a)
                # env.render() if render and i % 1 == 0 else None
                
                if (render and i % 1 == 0) or (i > 180 and np.mean(victories[-smoothing_num:]) < 0.001):
                    env.render()

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
            
            # print("\t\teps = {}; return = {}; expected return = {}".format(epsilon, episode_return, real_returns[starting_state_idx]))
            # print("\t{}: {}; {}/{}".format(i, starting_state_idx, episode_return, real_returns[starting_state_idx]))
            starting_return = real_returns[0] - real_returns[starting_state_idx]
            # print(epsilon)
            print("\t{}: {}; {}/{}".format(i, starting_state_idx, episode_return + starting_return, real_returns[0]))
            
            # TODO: save it in a dictionary (for example, based on reward or duration) or do it in post process
            # saving the seed(i) is necessary for replaying the episode later
            trajectories.append((current_trajectory, seed))

            losses.append(loss)
            episode_durations.append(duration)
            returns_trends.append(episode_return)
            
            dr = episode_return - real_returns[starting_state_idx]
            victory = dr >= - 0.1*abs(real_returns[starting_state_idx])
            
            victories.append(int(victory))
            
            # TODO - multiply it by gamma**len(trajectory till the starting point)
            disc_rewards.append(disc_reward)
            num_recent_victories = np.sum(victories[-smoothing_num:])
            print("\t\tNumber of Recent Victories ", num_recent_victories)
            
            # if len(victories) > smoothing_num and num_recent_victories >= stop_coeff:
            if len(victories) > smoothing_num and num_recent_victories >= stop_coeff:
            # if num_recent_victories >= stop_coeff:
                break
        
        print("Split", s, "finished in", i+1, "episodes out of ", max_num_episodes, ";", len(episode_durations), " episodes so far")
        
    return model, episode_durations, returns_trends, disc_rewards, losses, trajectories
