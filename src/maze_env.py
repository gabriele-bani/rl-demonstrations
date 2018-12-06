
import numpy as np
import sys
from gym import Env, spaces

from io import StringIO

import numpy
from numpy.random import random_integers as rand

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as cmap

COMPLEXITY = 1
DENSITY = 1


def build_maze(width=81, height=51, complexity=.75, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) # number of components
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) # size of components
    # Build actual maze
    Z = numpy.zeros(shape, dtype=int)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make islands
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2 # pick a random position
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    
    # MAKE SURE THE STARTING POINT AND THE ENDING POINT ARE FREE
    Z[1, 1] = Z[-2, -2] = 0
    
    return Z


UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

ACTIONS = [
    (-1, 0),    # UP
    (0, 1),     # RIGHT
    (1, 0),     # DOWN
    (0, -1),    # LEFT
]

cmaplist = [
    (1., 1., 1., 1.), # white floor
    (0., 0., 0., 1.), # black walls
    (0., 0., 1., 1.), # blue start
    (1., 0., 0., 1.), # red target
    (0., 1., 0., 1.), # green agent
]
colormap = cmap.from_list('maze', cmaplist, len(cmaplist))


class MazeEnv(Env):

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[10, 10], complexity=COMPLEXITY, density=DENSITY):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')


        self.maze = build_maze(shape[0], shape[1], complexity=complexity, density=density)
        
        shape = self.maze.shape
        
        self.shape = shape
        self.nS = np.prod(shape)
        self.nA = 4

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.start = (1, 1)
        self.end = (shape[0]-2, shape[1]-2)
        
        print(self.maze)
        
        assert self.maze[self.start] == 0
        assert self.maze[self.end] == 0

        super(MazeEnv, self).__init__()
        
        self.reset()

    def reset(self):
        self.s = self.start
        self.lastaction = None
        
        if hasattr(self, "_rendered_maze") and self._rendered_maze is not None:
            # self._rendered_maze.close()
            plt.close()
        
        self._rendered_maze = None
        
        return self.s

    def step(self, a):
        self.lastaction = a
        
        if self.s == self.end:
            return self.s, 0, True, {}
        
        dy, dx = ACTIONS[a]
        y, x = self.s
        
        x += dx
        y += dy
        
        s = y, x
        
        if self.maze[s] == 0:
            self.s = s
        
        d = self.s == self.end
        
        r = 0 if d else -1
        
        return self.s, r, d, {}

    def render(self, mode='human', close=False):
        if close:
            return
        
        if mode == "plot":
            maze = self.maze.copy()
            maze[self.start] = 2
            maze[self.end] = 3
            maze[self.s] = 4
            
            if self._rendered_maze is None:
                # self._rendered_maze = plt.figure(figsize=(10, 5))
                _, self._rendered_maze = plt.subplots(1, figsize=(10, 5))

            self._rendered_maze.cla()
            self._rendered_maze.imshow(maze, cmap=colormap, interpolation='nearest')
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.show()
            plt.pause(0.01)
        else:

            outfile = StringIO() if mode == 'ansi' else sys.stdout

            for y in range(self.maze.shape[0]):
                for x in range(self.maze.shape[1]):
                    
                    s = y, x
                    
                    if self.s == s:
                        output = " x "
                    elif s == self.start:
                        output = " S "
                    elif s == self.end:
                        output = " T "
                    elif self.maze[s] == 1:
                        output = " # "
                    else:
                        output = "   "

                    if x == 0:
                        output = output.lstrip()
                    if x == self.shape[1] - 1:
                        output = output.rstrip()

                    outfile.write(output)

                    if x == self.shape[1] - 1:
                        outfile.write("\n")

