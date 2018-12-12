import numpy as np
import matplotlib.pyplot as plt
from replay import play_trajectory
import utils
import random

from itertools import product


ROWS = 15
COLS = 15
SEED = 42
COMPLEXITY = 1
DENSITY = 1

discount_fact = 1.


def solve_maze_complete(maze_env):
    
    # run Floyd-Warshall algorithm to find the shortest paths between every pair of nodes
    maze = maze_env.maze
    
    R, C = maze.shape
    
    # initialize the distance to an high upperbound
    distance = R*C*np.ones((R, C, R, C), dtype=int)
    
    # build the matrix storing the optimal policy
    policy = -1*np.ones((R, C, R, C), dtype=int)
    
    for c in product(range(0, R), range(0, C)):
        if maze[c] != 1:
            distance[c + c] = 0
            policy[c + c] = -1
            
            # the end of the maze does not have any outgoing edge
            if c != maze_env.end:
                for a, (dy, dx) in enumerate(maze_env.ACTIONS):
                    # neighbor cell
                    n = c[0] + dy, c[1] + dx
                    if maze[n] != 1:
                        distance[c + n] = 1
                        policy[c + n] = a
    
    for k in product(range(0, R), range(0, C)):
        print(k[0], "/", R)
        if not maze[k]:
            for i in product(range(0, R), range(0, C)):
                if not maze[i]:
                    for j in product(range(0, R), range(0, C)):
                        if not maze[j]:
                            if distance[i + j] > distance[i + k] + distance[k + j]:
                                distance[i + j] = distance[i + k] + distance[k + j]
                                policy[i + j] = policy[i + k]
    
    return -1*distance, policy


def suboptimal_trajectory(maze_env, values, policy, percentile=0.8, render=False):
    
    def build_trajectory(maze_env, start, end, policy, render=False):
        done = False
        trajectory = []
        s = start
        while not done and s != end:
            a = policy[s + end]
            new_s, r, done, _ = maze_env.step(a)
            if render:
                maze_env.render("plot")
            trajectory.append((s, a, r, new_s, done))
            s = new_s
        return done, trajectory
    
    maze = maze_env.maze
    
    R, C = maze.shape
    
    start = maze_env.start
    end = maze_env.end
    
    distances = []

    for m in product(range(0, R), range(0, C)):
        if m != start and m != end and not maze[m]:
            walls = 0
            for dy, dx in maze_env.ACTIONS:
                walls += maze[m[0]+dy, m[1]+dx]
            # sample only nodes on a crossing
            if walls < 2:
                distances.append(m)
    
    
    
    trajectory = []
    s = maze_env.reset()
    done = False
    s = maze_env.reset()
    done = False
    
    for i in range(5):
        middle_node = random.choice(distances)
        print("Middle_node is", middle_node)
        
        done, t = build_trajectory(maze_env, s, middle_node, policy=policy, render=True)
        trajectory += t
        s = middle_node
        
        # print("Sub-optimal return: {}/{}".format(distance, values[start + end]))
        # print("Middle Node:", middle_node)
        # while not done and s != middle_node:
        #     a = policy[s + middle_node]
        #     new_s, r, done, _ = maze_env.step(a)
        #     if render:
        #         maze_env.render("plot")
        #     trajectory.append((s, a, r, new_s, done))
        #     s = new_s
    print("Final_node is", end)
    done, t = build_trajectory(maze_env, s, end, policy=policy, render=True)
    trajectory += t
    # while not done:
    #     a = policy[s + end]
    #     new_s, r, done, _ = maze_env.step(a)
    #     if render:
    #         maze_env.render("plot")
    #     trajectory.append((s, a, r, new_s, done))
    #     s = new_s
        
    return trajectory, done

def solve_maze(maze_env):
    # computes shortest paths from every cell to the end of the maze
    
    maze = maze_env.maze
    
    R, C = maze.shape
    
    # initialize the distance to an high upperbound
    distance = R*C*np.ones_like(maze)
    
    # build the matrix storing the optimal policy
    policy = -1*np.ones_like(maze, dtype=int)
    
    end = maze_env.end
    
    # run a BFS from the end
    
    distance[end] = 0
    policy[end] = 0
    queue = [end]
    
    while len(queue) > 0:
        
        # current cell
        c = queue.pop(0)
        
        for a, (dy, dx) in enumerate(maze_env.ACTIONS):
                # neighbor cell
                n = c[0] - dy, c[1] - dx
                
                # if it's not a wall and we haven't been there yet, append it to the queue
                if maze[n] == 0 and policy[n] < 0:
                    distance[n] = distance[c] + 1
                    policy[n] = a
                    queue.append(n)
    
    return -1*distance, policy

def epsilon_greedy_trajectory(maze_env, policy, epsilon=0.1, max_steps=None, render=False):
    
    if max_steps is None:
        max_steps = maze_env.shape[0] * maze_env.shape[1] / 2
        
    trajectory = []
    
    s = maze_env.reset()
    done = False
    
    while len(trajectory) < max_steps and not done:
        
        if random.uniform(0, 1) < epsilon:
            # print("Random Action")
            a = random.randint(0, maze_env.nA-1)
        else:
            a = policy[s[0], s[1]]
        new_s, r, done, _ = maze_env.step(a)
        
        if render:
            maze_env.render("plot")
        
        trajectory.append((s, a, r, new_s, done))
        
        s = new_s
    
    return trajectory, done
    

def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


env = utils.create_env("Maze", ROWS, COLS, SEED, COMPLEXITY, DENSITY)
env_name = env.get_name()


# distances, policy = solve_maze(env)
distances, policy = solve_maze_complete(env)

print("Solved!")

# A few tests
# for i in range(8):
#     # epsilon_greedy_trajectory(env, policy, epsilon=i * 0.1, render=True)
#     suboptimal_trajectory(env, distances, policy, percentile=1 - i * 0.1, render=True)


# Computing trajectories

trajectories = []
episode_durations = []
rewards = []
disc_rewards = []


for i in range(5):
# trajectory, done = epsilon_greedy_trajectory(env, policy, epsilon=0.9**i, render=False)
    trajectory, done = suboptimal_trajectory(env, distances, policy, percentile=0.95**0, render=False)
    # if done:
    trajectories.append((trajectory, SEED))
    
    episode_durations.append(len(trajectory))
    rewards.append(sum([r for _, _, r, _, _ in trajectory]))
    disc_rewards.append(sum([r*discount_fact**n for n, (_, _, r, _, _) in enumerate(trajectory)]))
    

d = {
    "rewards": rewards,
    "discounted rewards": disc_rewards,
    "episode durations": episode_durations
    }

utils.store_results(env_name, d)
utils.store_model(env_name, policy)
utils.store_trajectories(env_name, trajectories, None, 1.)

fig, axes = plt.subplots(nrows=1, ncols=3)
for i, (n, l) in enumerate(d.items()):
    print(i, n, l)
    axes[i].plot(smooth(l, 1))
    axes[i].set_title(n)
plt.show()


# play the first 20 trajectories.
# note that to reproduce exactly the trajectory, we have to also pass the seed (trajectories[i][1])
# so that we completely match the random effects that were present in the original run
print("start replaying trajectories")
for i in range(5):
    print("replaying trajectory", -i)
    play_trajectory(env, trajectories[-i][0], seed=trajectories[-i][1])
    input()
    play_trajectory(env, trajectories[-i][0], seed=trajectories[-i][1])
