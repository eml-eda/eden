""" 
Functions called on the output of the model to aggregate the results.
The predict_raw always outputs a 3D tensor, with shape (n_samples, n_trees, leaf_shape).
We aggregate over the n_trees.
"""
import numpy as np


def sum(x):
    return np.sum(x, axis=1)

def mean(x):
    return np.mean(x, axis=1)