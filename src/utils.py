
import envs
import gym
import pandas as pd
import numpy as np
import os
from datetime import datetime
import torch

import re

maze_parser = re.compile(r"""Maze_\((\d+),(\d+),(\d+),(\d*\.\d+|\d+),(\d*\.\d+|\d+)\)""")

import glob

DIRPATH = os.path.dirname(os.path.realpath(__file__))

DATADIR = f"{DIRPATH}/../data/"


def create_env(name):
    params = maze_parser.fullmatch(name)
    
    if params is not None:
        env = envs.MazeEnv(*params)
        assert env.get_name() == name, "Error! The parameters and the name of the environment given don't match!"
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
    file = f"{dir}/model.pt"
    torch.save(model, file)
    return file


def load_model(env_name):
    dir = build_data_dir(env_name)
    file = f"{dir}/model.pt"
    model = torch.load(file)
    return model


def store_results(env_name, results):
    dir = build_data_dir(env_name)
    file = f"{dir}/results.pkl"
    torch.save(results, file)
    return file


def load_results(env_name):
    dir = build_data_dir(env_name)
    file = f"{dir}/results.pkl"
    results = torch.load(file)
    return results


def trajetory_to_dataframe(env_name, trajectories, env_params, discount, **kwargs):
    dataframe = []
    
    if env_params == None:
        env_params = {}
    
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


def store_trajectories(env_name, trajectories, env_params, discount, **kwargs):
    df = trajetory_to_dataframe(env_name, trajectories, env_params, discount, **kwargs)
    
    outdir = build_data_dir(env_name)
    
    id = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    
    outfile = os.path.join(outdir, "trajectories_({})".format(id))
    df.to_pickle(outfile)


def load_trajectories(env_name, filename=None):
    
    if filename is None:
        
        dir = build_data_dir(env_name)
        files = glob.glob(os.path.join(dir, "trajectories_*"))
        
        # pick the most recent one
        files = sorted(files)
        filename = files[-1]
    
    df = pd.read_pickle(filename)
    
    return df
