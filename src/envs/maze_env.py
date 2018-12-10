
import numpy as np
import sys
from gym import Env, spaces

from io import StringIO

import numpy
import random

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as cmap

COMPLEXITY = 1.
DENSITY = 1.


def build_maze(width=81, height=51, complexity=.75, density=.75, seed=42):
    
    # local random number generator using the seed specified
    rng = random.Random(seed)
    
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) # number of components
    density = int(density * ((shape[0] // 2) * (shape[1] // 2))) # size of components
    # Build actual maze
    Z = numpy.zeros(shape, dtype=int)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make islands
    for i in range(density):
        x, y = rng.randint(0, shape[1] // 2) * 2, rng.randint(0, shape[0] // 2) * 2 # pick a random position
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
                y_,x_ = neighbours[rng.randint(0, len(neighbours) - 1)]
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

# ACTIONS = [
#     np.array([-1, 0]),    # UP
#     np.array([0, 1]),     # RIGHT
#     np.array([1, 0]),     # DOWN
#     np.array([0, -1]),    # LEFT
# ]
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

    def __init__(self, rows=10, cols=10, maze_seed=42, complexity=COMPLEXITY, density=DENSITY):

        super(MazeEnv, self).__init__()
        
        shape = [2*s+1 for s in [rows, cols]]

        self._maze_seed = maze_seed
        self.complexity = complexity
        self.density = density
        self.shape = shape
        
        self.nS = np.prod(shape)
        self.nA = 4
        self.ACTIONS = ACTIONS

        self.maze = build_maze(self.shape[0],
                               self.shape[1],
                               complexity=self.complexity,
                               density=self.density,
                               seed=self._maze_seed)
        
        self.s = None
        self.lastaction = None
        self._rendered_maze = None

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        
        self.start = (1, 1)
        self.end = (shape[0]-2, shape[1]-2)
        
        # self.seed(seed)
        self.reset()
        
    # def seed(self, seed=None):
    #     if seed is not None:
    #         self._rndseed = seed
    #     return self._rndseed
    
    def get_params(self):
        return (self.shape[0]//2,
                self.shape[1]//2,
                self._maze_seed,
                float(self.complexity),
                float(self.density))
    
    def get_name(self):
        return "Maze_({},{},{},{},{})".format(*self.get_params())
    
    def reset(self):
        
        # self.maze = build_maze(self.shape[0],
        #                        self.shape[1],
        #                        complexity=self.complexity,
        #                        density=self.density,
        #                        seed=self.seed())
    
        assert self.maze[self.start] == 0
        assert self.maze[self.end] == 0
        
        self.s = self.start
        self.lastaction = None
        
        if self._rendered_maze is not None:
            plt.close(self._rendered_maze[0])
        
        self._rendered_maze = None
        
        # self.render()
        
        return self.s

    def step(self, a):
        self.lastaction = a
        
        if self.s == self.end:
            return self.s, 0, True, {}
        
        # print("action", a)
        dy, dx = self.ACTIONS[a]
        y, x = self.s

        x += dx
        y += dy
        
        s = y, x
        
        if self.maze[s] == 0:
            self.s = s
        
        d = self.s == self.end
        
        r = 0 if d else -1
        
        return self.s, r, d, {}

    def render(self, mode='plot', close=False):
        
        if close:
            return
        
        if mode == "plot":
            maze = self.maze.copy()
            maze[self.start] = 2
            maze[self.end] = 3
            maze[self.s] = 4
            
            if self._rendered_maze is None:
                self._rendered_maze = plt.subplots(1, figsize=(10, 5))

            self._rendered_maze[1].cla()
            self._rendered_maze[1].imshow(maze, cmap=colormap, interpolation='nearest')
            self._rendered_maze[1].set_xticks([])
            self._rendered_maze[1].set_yticks([])
            self._rendered_maze[0].canvas.draw_idle()
            self._rendered_maze[0].show()
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

