import numpy as np
import utils
# from itertools import zip_longest
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

import os.path

# plt.xkcd()

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def build_plot(env_name, selection_conditions: Dict =None, MAXx=None, MINy=None, MAXy=None, smooth=10):
    
    w = 1
    
    # experiments = []
    # for d in [0, 100, 500]:
    #     experiments += [
    #         {
    #             "returns": [d + round(200 -i) + np.random.randint(-50, 50) for i in range(1, np.random.randint(50, 180))],
    #             "seed": np.random.randint(0, 2000),
    #             "demonstration_value": d,
    #             "chunks": np.random.choice([0, 1, 2, 3, 4]),
    #             "eps_iterations": np.random.choice([0, 1, 2, 3, 4, 8, 10, 17])
    #         }
    #         for j in range(50)
    #     ]
    #
    # experiments = pd.DataFrame(experiments)

    experiments = utils.load_experiments(env_name)
    
    if selection_conditions is not None:
        for column, values in selection_conditions.items():
            experiments = experiments[experiments[column].isin(values)]
    
    experiments.demonstration_value.fillna("From Scratch", inplace=True)
    
    experiments["sortkey"] = experiments.demonstration_value.apply(lambda x: -x if not isinstance(x, str) else -10000)
    experiments.sort_values(by="sortkey", inplace=True)
    
    maps = pd.DataFrame({"demonstration_value": experiments.demonstration_value.unique(),
                         "color": ["red", "green", "orange", "blue"],
                         "label": ["From Scratch", "Optimal Demo", "Suboptimal Demo", "Bad Demo"]}
                        )
    print(maps)
    
    experiments = experiments.merge(maps, on="demonstration_value")
    
    plt.figure()
    
    best_demostration = -100000
    longest_train = 1
    
    for i, row in experiments.iterrows():
        
        if not row.label in ["Optimal Demo"]:
            continue
            
        try:
            v = float(row.demonstration_value)
            best_demostration = max(v, best_demostration)
        except ValueError:
            pass

        returns = row.returns
        length = len(returns) + 1
        
        returns += [returns[-1]]*(750 - length)
        length = len(returns) + 1
        
        longest_train = max(longest_train, length)

        if row.label != "From Scratch":
            label = "{} (G={})".format(row.label, row.demonstration_value)
        else:
            label = row.label
        
        plt.plot(np.arange(smooth, length), utils.smooth(returns, smooth), label=label, color=row.color, alpha=0.1)

    # plt.legend(title="Demonstration Value")
    # plt.legend()
    plt.title("Episodes' Returns during Training")
    plt.xlabel("Episode during training")
    plt.ylabel("Total Return in episode")

    if MINy is not None:
        plt.ylim(ymin=MINy)
        
    if MAXy is not None:
        plt.ylim(ymax=MAXy)

    if MAXx is not None:
        longest_train = min(MAXx, longest_train)
        plt.xlim(0, MAXx)

    # ymin, ymax = plt.ylim()
    # for i, row in statistics.iterrows():
    #     color = colors[row.demonstration_value]
    #     plt.vlines(row.length_mean, ymin=ymin, ymax=ymax, linewidth=0.5, color=color)
    #     plt.axvspan(row.length_mean - w*row.length_std, row.length_mean + w*row.length_std, color=color, alpha=0.03)
        
    plt.hlines(best_demostration, xmin=0, xmax=longest_train, linewidth=0.5, color='black', linestyles='--', label='optimal trajectory')
    plt.yticks(list(plt.yticks()[0]) + [best_demostration])
    
    dir = utils.build_data_dir(env_name)
    outfile = os.path.join(dir, "{}.svg".format(env_name))
    
    plt.draw()
    # plt.savefig(outfile, format='svg', dpi=1000)
    plt.show()


build_plot("Maze_(15,15,42,1.0,1.0)", {"chunks": [5, None], "eps_iterations": [10, None]}, MAXx=None, MINy=-1500, MAXy=0)

build_plot("MountainCar-v0", {"chunks": [20, None], "eps_iterations": [100, None]}, MAXx=3750)

