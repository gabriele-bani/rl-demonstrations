import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from tqdm import tqdm as _tqdm
import random
from train_QNet import *
import copy



# @profile
def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, get_epsilon,
                 use_target_qnet=None, render=False):

    optimizer = optim.Adam(model.parameters(), learn_rate)
    if use_target_qnet != None:
        target_model = copy.deepcopy(model)
    else:
        target_model, target_optim = None, None

    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    rewards = []
    disc_rewards = []
    losses = []
    trajectories = []
    # ys = []
    # xs = []


    for i in tqdm(range(num_episodes)):
        duration = 0
        reward = 0
        disc_reward = 0
        loss = -99999

        trajectory = []
        # max_y, max_x = -100, -100

        if use_target_qnet != None and i % 5 == 0:
            target_model = copy.deepcopy(model)

        random.seed(i)
        torch.manual_seed(i)
        env.seed(i)

        s = env.reset()
        env.render() if render and i % 1 == 0 else None

        while True:
            epsilon = get_epsilon(i)
            a = select_action(model, s, epsilon)

            next_state, r, done, _ = env.step(a)
            env.render() if render and i % 1 == 0 else None

            # max_x = max(max_x, next_state[0])
            # max_y = max(max_y, next_state[1])
            # r = next_state[0] if r == -1 else 200

            duration += 1
            reward += r

            disc_reward += (discount_factor ** duration) * r

            trajectory.append((s, a, r, next_state, done))
            memory.push((s, a, r, next_state, done))
            loss = train(model, memory, optimizer, batch_size, discount_factor,
                         target_model=target_model)
            global_steps += 1

            if done:
                break

            s = next_state

        # xs.append(max_x)
        # ys.append(max_y)
        # TODO: save it in a dictionary (for example, based on reward or duration) or do it in post process
        # saving the seed(i) is necessary for replaying the episode later
        trajectories.append((trajectory, i))

        losses.append(loss)
        episode_durations.append(duration)
        rewards.append(reward)
        disc_rewards.append(disc_reward)

    return episode_durations, rewards, disc_rewards, losses, trajectories