import numpy as np
import utils
# from itertools import zip_longest
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict
import os

# plt.xkcd()


pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def build_plot(env_name, selection_conditions: Dict =None, target_column = "train_length"):

    w = 1
    
    experiments = utils.load_experiments(env_name)
    
    experiments["return"] = experiments.returns.apply(lambda x: x[-1])
    
    print(experiments)
    
    from_scratch = experiments[experiments.demonstration_value.isnull()][target_column].mean()
    
    if selection_conditions is not None:
        for column, values in selection_conditions.items():
            experiments = experiments[experiments[column].isin(values)]
    
    statistics = experiments.groupby(by=["chunks", "eps_iterations"])[target_column].agg(["mean", "std"]).reset_index()
    
    plt.figure()
    
    xmin = 1000
    xmax = 0
    
    for eps_iterations, group in statistics.groupby("eps_iterations"):
        chunks = group["chunks"]
        returns_mean, returns_std = group["mean"], group["std"]
        
        xmin = min(xmin, chunks.min())
        xmax = max(xmax, chunks.max())
        
        plt.plot(chunks, returns_mean, label=eps_iterations)
        plt.fill_between(chunks, returns_mean - w*returns_std, returns_mean + w*returns_std, alpha=0.2)
    
    plt.hlines(from_scratch, xmin=xmin, xmax=xmax, linewidth=0.5, color='black', linestyles='--', label='Train From Scratch')
    plt.yticks(list(plt.yticks()[0]) + [from_scratch])

    plt.legend(title="Epsilon Iterations")
    # plt.title("Training Hyperparameters Study")
    
    if target_column == "return":
        ylabel = "Test Returns"
    elif target_column == "train_length":
        ylabel = "Number of Episodes seen during Training"
    else:
        ylabel = "UNKNOWN"
    
    plt.xlabel("Number of Splits in Training")
    plt.ylabel(ylabel)
    
    dir = utils.build_data_dir(env_name)
    outfile = os.path.join(dir, "{}_({})_hyperparams.svg".format(env_name, target_column))

    plt.draw()
    plt.savefig(outfile, format='svg', dpi=1000)
    plt.show()


# build_plot("LunarLander-v2", target_column="train_length")
# build_plot("LunarLander-v2", target_column="return")
#
build_plot("Maze_(15,15,42,1.0,1.0)", {"demonstration_value": [-67]}, target_column="return")
build_plot("Maze_(15,15,42,1.0,1.0)", {"demonstration_value": [-67]}, target_column="train_length")
#
# build_plot("MountainCar-v0", {"demonstration_value": [-87]}, target_column="return")
# build_plot("MountainCar-v0", {"demonstration_value": [-87]}, target_column="train_length")
