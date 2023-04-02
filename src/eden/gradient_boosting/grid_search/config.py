"""
File storing all the configurations and common functions for the grid searches
of each dataset
"""
import numpy as np
import itertools
import os

os.makedirs("logs/gbdt/ninapro", exist_ok=True)
os.makedirs("logs/gbdt/unimib", exist_ok=True)
os.makedirs("logs/gbdt/ecg5000", exist_ok=True)
os.makedirs("logs/gbdt/infrared", exist_ok=True)
os.makedirs("logs/gbdt/tuar", exist_ok=True)


GBT_BASE_PARAMS_GRID = {"max_depth": np.arange(1, 15), "input_bits": [None, 32, 16, 8]}
PARAMS = [*itertools.product(*[v for _, v in GBT_BASE_PARAMS_GRID.items()])][::-1]
LEAVES_BITS = [None, 32, 16, 8]
GBT_MAX_ESTIMATORS = 40
