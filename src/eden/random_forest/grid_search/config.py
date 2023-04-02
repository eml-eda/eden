"""
File storing all the configurations and common functions for the grid searches
of each dataset
"""
import numpy as np
import itertools

import os

os.makedirs("logs/rf/ninapro", exist_ok=True)
os.makedirs("logs/rf/unimib", exist_ok=True)
os.makedirs("logs/rf/ecg5000", exist_ok=True)
os.makedirs("logs/rf/tuar", exist_ok=True)
os.makedirs("logs/rf/hdd", exist_ok=True)

RF_BASE_PARAMS_GRID = {"max_depth": np.arange(1, 15), "input_bits": [32, 16, 8]}
PARAMS = [*itertools.product(*[v for _, v in RF_BASE_PARAMS_GRID.items()])]
LEAVES_BITS = [32, 16, 8]
RF_MAX_ESTIMATORS = 40
