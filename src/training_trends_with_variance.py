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


def zip_longest(*lists):
    def g(l):
        for item in l:
            yield item
        while True:
            yield l[-1]
    gens = [g(l) for l in lists]
    for _ in range(max(map(len, lists))):
        yield tuple(next(g) for g in gens)


def build_line(group):
    num_samples = len(group)
    returns_mean = []
    returns_std = []

    length_mean = group.train_length.mean()
    length_std = group.train_length.std()

    for i, samples in enumerate(zip_longest(*group.returns)):
        samples = [s for s in samples if s is not None]
    
        mean = np.mean(samples)
        std = np.std(samples)
    
        returns_mean.append(mean)
        returns_std.append(std)

    returns_mean = np.array(returns_mean)
    returns_std = np.array(returns_std)
    return pd.DataFrame([{"returns_mean": returns_mean, "returns_std": returns_std, "length_mean": length_mean, "length_std": length_std, "num_samples": num_samples}])


def build_plot(env_name, selection_conditions: Dict =None, MAXx=None, MINy=None, MAXy=None, smooth=5):
    
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

    print(experiments[["demonstration_value", "chunks", "eps_iterations"]])
    
    if selection_conditions is not None:
        for column, values in selection_conditions.items():
            experiments = experiments[experiments[column].isin(values)]
    
    experiments.demonstration_value.fillna("From Scratch", inplace=True)
    
    experiments["short"] = experiments.returns.apply(lambda x: x[-3:])
    # print(experiments[["demonstration_value", "seed", "train_length", "short"]])
    

    # for col in experiments:
    #     print(col)
    #     if col not in ["trajectories", "returns", "short"]:
    #         print(experiments[col].unique())
    #
    # experiments = experiments[experiments.demonstration_value == -199.0]
    # print(experiments.groupby(["seed", "demonstration_value"]).size())
    
    statistics = experiments.groupby(by="demonstration_value").apply(build_line).reset_index()
    statistics["sortkey"] = statistics.demonstration_value.apply(lambda x: -x if not isinstance(x, str) else -10000)
    statistics.sort_values(by="sortkey", inplace=True)
    
    print(statistics)
    
    statistics["color"] = ["red", "green", "orange", "blue"]
    statistics["label"] = ["From Scratch", "Optimal Demo", "Suboptimal Demo", "Bad Demo"]
    
    plt.figure()
    
    best_demostration = -100000
    longest_train = 1
    
    for i, row in statistics.iterrows():
        
        try:
            v = float(row.demonstration_value)
            best_demostration = max(v, best_demostration)
        except ValueError:
            pass

        returns_mean, returns_std = row.returns_mean, row.returns_std
        
        print("Num samples for {}: {}".format(row.demonstration_value, row.num_samples))
        
        # length = returns_mean.shape[0] - smooth + 1
        length = returns_mean.shape[0] + 1
        longest_train = max(longest_train, length)

        if row.label != "From Scratch":
            label = "{} (G={})".format(row.label, row.demonstration_value)
        else:
            label = row.label
        
        plt.plot(np.arange(smooth, length), utils.smooth(returns_mean, smooth), label=label, color=row.color)
        plt.fill_between(np.arange(smooth, length),
                         utils.smooth(returns_mean - w*returns_std, smooth),
                         utils.smooth(returns_mean + w*returns_std, smooth),
                         alpha=0.1, color=row.color)

    # plt.legend(title="Demonstration Value")
    plt.legend()
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
    plt.savefig(outfile, format='svg', dpi=1000)
    plt.show()


# build_plot("Maze_(15,15,42,1.0,1.0)", {"chunks": [5, None], "eps_iterations": [10, None]}, MAXx=None, MINy=-1500, MAXy=0)
#
# build_plot("MountainCar-v0", {"chunks": [15, None], "eps_iterations": [20, None]}, MAXx=600)

build_plot("LunarLander-v2", {"chunks": [30, None], "eps_iterations": [20, None]}, MAXx=1100, MINy=-400)
