import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from train_QNet import select_action
import gym

import utils

###
### the functions in this file have the only purpose of visualization
###
frame_time = 0.02


def play_episodes(env, model, n=20, seed=42, render=True):
    episode_durations = []

    assert seed == int(seed)
    seed = int(seed)
    
    for i in range(n):
        env.seed(seed + i)
        state = env.reset()
        
        if render:
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

            if render:
                env.render()
                time.sleep(frame_time)

        episode_durations.append(steps)
        env.close()

    plt.plot(episode_durations)
    plt.title('Episode durations')
    plt.show()


# trajectories are in the form [(s, a, r, s'), ...]
def play_trajectory(env, trajectory, seed=42, render=True):
    episode_durations = []

    assert seed == int(seed)
    seed = int(seed)
    
    random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    state = env.reset()
    j = 0

    if render:
        env.render()
    time.sleep(frame_time)

    done = False
    steps = 0

    while not done:
        steps += 1

        action = trajectory[j][1]
        state, reward, done, _ = env.step(action)

        if np.mean(np.abs(np.array(state) - np.array(trajectory[j][3]))) > 1e-15:
            print(state, trajectory[j][3])
            print("the trajectory and the simulation do not match! watch the seeds!")
            raise ValueError

        if render:
            env.render()
            time.sleep(frame_time)
        j += 1

    episode_durations.append(steps)
    env.close()


if __name__ == "__main__":

    env_name = "MountainCar-v0"
    
    # env_name = "Maze_(15,15,42,1.0,1.0)"
    
    env = utils.create_env(env_name)
    
    model = utils.load_model(env_name)
    trajectories = utils.load_trajectories(env_name)
    d = utils.load_results(env_name)
    
    # TODO - FIX IN CASE OF MAZE, DEPENDING ON HOW WE IMPLEMENT THE MODEL
    print("start playing episodes with the trained model")
    play_episodes(env, model, 3)
    
    c = 0
    print("start replaying trajectories")
    for i, row in trajectories.iloc[::-1].iterrows():
        print("replaying trajectory", -i)
        
        play_trajectory(env, row["trajectory"], seed=row["seed"])
        
        c += 1
        if c > 5:
            break
