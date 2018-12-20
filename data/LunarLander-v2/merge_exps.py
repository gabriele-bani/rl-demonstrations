#!/usr/bin/python

import pandas as pd

scratch = pd.read_pickle("scratch.pkl")

exp = pd.read_pickle("experiments.pkl")

gabexp = pd.read_pickle("gabriexp.pkl")


exp = pd.concat([scratch, exp, gabexp])

exp.to_pickle("exp.pkl")
