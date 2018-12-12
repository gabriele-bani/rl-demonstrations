import numpy as np
import utils
from itertools import zip_longest
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict

plt.xkcd()

env_name = "MountainCar-v0"


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
    
    experiments = []
    for d in [0, 100, 500]:
        experiments += [
            {
                "returns": [d + round(200 -i) + np.random.randint(-50, 50) for i in range(1, np.random.randint(50, 180))],
                "seed": np.random.randint(0, 2000),
                "demonstration_value": d,
                "chunks": np.random.choice([0, 1, 2, 3, 4]),
                "eps_iterations": np.random.choice([0, 1, 2, 3, 4, 8, 10, 17])
            }
            for j in range(50)
        ]

    experiments = pd.DataFrame(experiments)

    # experiments = utils.load_experiments(env_name)
    
    if selection_conditions is not None:
        for column, value in selection_conditions.items():
            experiments = experiments[experiments[column] == value]
    
    statistics = experiments.groupby(by="demonstration_value").apply(build_line).reset_index()
    # print(statistics)

    plt.figure()

    for i, row in statistics.iterrows():
        returns_mean, returns_std = row.returns_mean, row.returns_std
        plt.plot(returns_mean, label=row.demonstration_value)
        plt.fill_between(np.arange(0, returns_mean.shape[0]), returns_mean - w*returns_std, returns_mean + w*returns_std)
    plt.legend(title="Demonstration Value")
    plt.title("Episodes' Returns during Training")
    plt.xlabel("Episode during training")
    plt.ylabel("Total Return in episode")
    plt.show()


build_plot(env_name, {"chunks": 4, "eps_iterations": 17})