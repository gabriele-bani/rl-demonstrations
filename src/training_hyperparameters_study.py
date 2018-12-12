import numpy as np
import utils
from itertools import zip_longest
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

# plt.xkcd()

# env_name = "MountainCar-v0"

env_name = "Maze_(15,15,42,1.0,1.0)"


# row["env_name"] = env_name
# row["env_params"] = env_params
# row["returns"] = returns
# row["seed"] = seed
# row["demonstration_value"] = demonstration_value
# row["chunks"] = chunks
# row["eps_iterations"] = eps_iterations
# row["stop_victories"] = stop_victories
# row["smoothing_victories"] = smoothing_victories
# row["train_length"] = len(returns)
# row["time"] = time

def build_line(group):
    returns_mean = []
    returns_std = []
    for i, samples in enumerate(zip_longest(*group.returns)):
        samples = [s for s in samples if s is not None]
        
        mean = np.mean(samples)
        std = np.std(samples)
        
        returns_mean.append(mean)
        returns_std.append(std)

    returns_mean = np.array(returns_mean)
    returns_std = np.array(returns_std)
    return pd.DataFrame([{"returns_mean": returns_mean, "returns_std": returns_std}]) #, "demonstration_value": group.iloc[0].demonstration_value}])


def build_plot(env_name, selection_conditions: Dict =None, w=0.5):
    
    # experiments = []
    # for d in [0, 100, 500]:
    #     for c in [0, 1, 2, 3, 4]:
    #         experiments += [
    #             {
    #                 "returns": [d + round(200 -i) + np.random.randint(-50, 50) for i in range(1, np.random.randint(160, 180))],
    #                 "seed": np.random.randint(0, 2000),
    #                 "demonstration_value": d,
    #                 "chunks": c,
    #                 "eps_iterations": np.random.choice([0, 1, 5, 10, 50])
    #             }
    #             for j in range(50)
    #         ]
    # for i in range(len(experiments)):
    #     experiments[i]["train_length"] = len(experiments[i]["returns"])
    #
    # experiments = pd.DataFrame(experiments)

    experiments = utils.load_experiments(env_name)

    if selection_conditions is not None:
        for column, values in selection_conditions.items():
            experiments = experiments[experiments[column].isin(values)]
    
    statistics = experiments.groupby(by=["chunks", "eps_iterations"]).train_length.agg(["mean", "std"]).reset_index()
    
    plt.figure()

    for eps_iterations, group in statistics.groupby("eps_iterations"):
        chunks = group["chunks"]
        returns_mean, returns_std = group["mean"], group["std"]
        plt.plot(chunks, returns_mean, label=eps_iterations)
        plt.fill_between(chunks, returns_mean - w*returns_std, returns_mean + w*returns_std, alpha=0.2)
    
    plt.legend(title="Epsilon Iterations")
    plt.title("Training Hyperparameters Study")
    plt.xlabel("Number of Splits in Training")
    plt.ylabel("Number of Episodes seen during Training")
    plt.show()


build_plot(env_name, {"demonstration_value": [-67]})
