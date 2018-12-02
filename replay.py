import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from train_functions import select_action
###
### the functions in this file have the only purpose of visualization
###
frame_time = 0.0025
seed = 42

def play_episodes(env, model, n=20):
    episode_durations = []
    for i in range(n):
        env.seed(seed + i)
        state = env.reset()

        env.render()
        time.sleep(frame_time)

        done = False
        steps = 0

        while not done:
            steps += 1

            with torch.no_grad():
                # action = bonus_get_action(state).item()
                action = select_action(model, state, epsilon=0)
            state, reward, done, _ = env.step(action)

            env.render()
            time.sleep(frame_time)

        episode_durations.append(steps)
        env.close()

    plt.plot(episode_durations)
    plt.title('Episode durations')
    plt.show()


# trajectories are in the form [(s, a, r, s'), ...]
def play_trajectory(env, trajectory, seed=42):
    episode_durations = []

    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    state = env.reset()
    j = 0

    env.render()
    time.sleep(frame_time)

    done = False
    steps = 0

    while not done:
        steps += 1

        action = trajectory[j][1]
        state, reward, done, _ = env.step(action)

        if np.mean(np.abs(state - trajectory[j][3])) > 1e-8:
            print(state, trajectory[j][3])
            raise ValueError

        env.render()
        time.sleep(frame_time)
        j += 1

    episode_durations.append(steps)
    env.close()

