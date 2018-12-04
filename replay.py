import torch
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import random
from train_QNet import select_action
import gym

###
### the functions in this file have the only purpose of visualization
###
frame_time = 0.005


def play_episodes(env, model, n=20, seed=42):
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
            print("the trajectory and the simulation do not match! watch the seeds!")
            raise ValueError

        env.render()
        time.sleep(frame_time)
        j += 1

    episode_durations.append(steps)
    env.close()


if __name__ == "__main__":

    env_name = "MountainCar-v0"

    model = torch.load(f"bin/{env_name}/weights.pt")
    d = torch.load(f"bin/{env_name}/results.pkl")
    trajectories = torch.load(f"bin/{env_name}/trajectories.pkl")

    env = gym.envs.make(env_name)

    print("start playing episodes with the trained model")
    play_episodes(env, model, 3)

    print("start replaying trajectories")
    for i in range(5):
        print("replaying trajectory", -i)
        play_trajectory(env, trajectories[-i][0], seed=trajectories[-i][1], )
