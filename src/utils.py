
import envs
import gym
import pandas as pd
import numpy as np
import os
from datetime import datetime
import torch
import math
import re
from typing import List, Tuple

maze_parser = re.compile(r"""Maze_\((\d+),(\d+),(\d+),(\d*\.\d+|\d+),(\d*\.\d+|\d+)\)""")

import glob

DIRPATH = os.path.dirname(os.path.realpath(__file__))

DATADIR = f"{DIRPATH}/../data/"


def smooth(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def create_env(name, *args, **kwargs):
    
    match = maze_parser.fullmatch(name)
    
    if name == "Maze":
        env = envs.MazeEnv(*args, **kwargs)
        return env
    elif match is not None:
        params = match.groups()
        env = envs.MazeEnv(int(params[0]), int(params[1]), int(params[2]), float(params[3]), float(params[4]))
        assert env.get_name() == name, "Error! The name generated doesn't match the name given: {} != {}".format(env.get_name(), name)
        return env
    else:
        return gym.envs.make(name)
    

def build_data_dir(env_name):
    dir = os.path.join(DATADIR, env_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir


def store_model(env_name, model):
    dir = build_data_dir(env_name)
    id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    file = os.path.join(dir, "model_({}).pt".format(id))
    
    torch.save(model, file)
    return file


def load_model(env_name, date=None):
    
    dir = build_data_dir(env_name)
    
    if date is None:
        
        dir = build_data_dir(env_name)
        files = glob.glob(os.path.join(dir, "model_(*).pt"))
        
        # pick the most recent one
        files = sorted(files)
        filename = files[-1]
    else:
        filename = os.path.join(dir, "model_({}).pt".format(date))
    
    model = torch.load(filename)
    return model


def store_results(env_name, results):
    dir = build_data_dir(env_name)
    id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    file = os.path.join(dir, "results_({}).pkl".format(id))

    torch.save(results, file)
    return file


def load_results(env_name, date=None):
    
    dir = build_data_dir(env_name)
    
    if date is None:
        
        dir = build_data_dir(env_name)
        files = glob.glob(os.path.join(dir, "results_(*).pkl"))
        
        # pick the most recent one
        files = sorted(files)
        filename = files[-1]
    else:
        filename = os.path.join(dir, "results_({}).pkl".format(date))
        
    results = torch.load(filename)
    return results


def trajectory_to_dataframe(env_name, trajectories, env_params, discount, **kwargs):
    dataframe = []
    
    for i, trajectory_tuple in enumerate(trajectories):
        trajectory, seed = trajectory_tuple
        
        row = {}
        
        row["seed"] = seed
        row["env_name"] = env_name
        row["trajectory"] = trajectory
        row["episode_length"] = len(trajectory)
        row["env_params"] = env_params
        # row["finished"] = finished
        row.update(kwargs)
        
        # Not really a way to account for the agent finishing right in the last second
        # trajectory_dict[i]["is_finished"] = len(trajectory) < env._max_episode_steps
        
        _, _, rewards, _, _ = zip(*trajectory)
        
        row["max_reward"] = np.max(rewards)
        row["sum_reward"] = np.sum(rewards)
        discounts = [np.power(discount, i) for i in range(len(rewards))][::-1]
        row["sum_discounted_reward"] = np.sum([rewards[i] * discounts[i] for i in range(len(rewards))])
        
        dataframe.append(row)
    
    df = pd.DataFrame(dataframe)
    
    return df


def store_trajectories(env_name, trajectories, env_params, discount, filename=None, **kwargs):
    df = trajectory_to_dataframe(env_name, trajectories, env_params, discount, **kwargs)
    
    outdir = build_data_dir(env_name)
    if filename is None:
        id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        
        outfile = os.path.join(outdir, "trajectories_({}).pkl".format(id))
    else:
        outfile = os.path.join(outdir, "{}.pkl".format(filename))

    df.to_pickle(outfile)


def load_trajectories(env_name, date=None, filename=None):
    dir = build_data_dir(env_name)
    if date is None and filename is None:

        files = glob.glob(os.path.join(dir, "trajectories_(*).pkl"))
        
        # pick the most recent one
        files = sorted(files)
        filename = files[-1]
    elif filename is None:
        filename = os.path.join(dir, "trajectories_({}).pkl".format(date))
    else:
        filename = os.path.join(dir, "{}.pkl".format(filename))
    
    df = pd.read_pickle(filename)
    
    return df


def chunks(l, n):
    
    assert n < l/2
    
    s = l // n
    r = l % n
    
    p = l
    
    while p > 0:
        yield list(range(p - s - (r > 0), p))
        p -= s + (r > 0)
        r -= 1


def experiments_to_dataframe(env_name, env_params, experiments: List[Tuple[List[int], int, int, int, int, int, int, List[int]]]):
    dataframe = []
    
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    
    for returns, seed, demonstration_value, chunks, eps_iterations, stop_victories, smoothing_victories, tests in experiments:
        row = {}
        
        row["env_name"] = env_name
        row["env_params"] = env_params
        
        # list of returns of each episode seen during training
        row["returns"] = returns
        # seed of the experiment
        row["seed"] = seed
        # return of the trajectory used as demonstration (None if trained from scratch, i.e. no demonstrations provided)
        row["demonstration_value"] = demonstration_value
        # number of chunks the trajectory has been split in (None if trained from scratch, i.e. no demonstrations provided)
        row["chunks"] = chunks
        # number of steps where the epsilon decreased linerly
        row["eps_iterations"] = eps_iterations
        # number of victories required for considering a split solved (None if trained from scratch, i.e. no demonstrations provided)
        row["stop_victories"] = stop_victories
        # number of episodes to consider to count the victories (None if trained from scratch, i.e. no demonstrations provided)
        row["smoothing_victories"] = smoothing_victories

        row["tests"] = tests
        
        row["train_length"] = len(returns)

        row["time"] = time

        
        
        dataframe.append(row)
    
    df = pd.DataFrame(dataframe)
    
    return df


def store_experiments(env_name: str, env_params, experiments: List[Tuple[List[int], int, int, int, int, int, int, List[int]]], filename: str = None):
    df = experiments_to_dataframe(env_name, env_params, experiments)
    
    dir = build_data_dir(env_name)
    
    if filename is None:
        filename = "experiments"
    
    fullpath = os.path.join(dir, "{}.pkl".format(filename))
    
    if os.path.isfile(fullpath):
        old_df = load_experiments(env_name, filename=filename)
        df = pd.concat([old_df, df], ignore_index=True)
        
    df.to_pickle(fullpath)
    
    return df


def load_experiments(env_name, filename=None):
    dir = build_data_dir(env_name)
    if filename is None:
        filename = os.path.join(dir, "experiments.pkl")
    else:
        filename = os.path.join(dir, "{}.pkl".format(filename))
    
    df = pd.read_pickle(filename)
    
    return df

