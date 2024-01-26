import numpy as np
from typing import Literal, Iterable, Tuple
import logging
from copy import deepcopy
from bigtree import preorder_iter
from eden.utils import nptype_from_name


def quantize(
    *,
    data: np.ndarray,
    min_val: float,
    max_val: float,
    precision: Literal[8, 16, 32] = 32,
    method: str = "clip",
) -> Tuple[np.ndarray, float, float]:
    """
    Base quantization function for alphas and inputs.

    Parameters
    ----------
    data : np.ndarray
        Data to be quantized
    min_val , max_val : float
        Range of the data to be quantized
    precision : Literal[8,16,32]
        Bit width desired.
    method : str
        Function to apply after scale and zero point, default is clip

    Returns
    -------
    np.ndarray
        A quantized copy of data
    float
        Scale factor
    float
        zero point
    """
    data = np.copy(data)
    qmin = 0
    qmax = 2 ** (precision) - 1
    scale = (max_val - min_val) / (2**precision - 1)
    zero_point = -(round(min_val / scale) - qmin)
    if method == "clip":
        data = np.round(data / scale + zero_point).astype(int)
    else:
        data = np.trunc(data / scale + zero_point).astype(int)
    data = np.clip(a=data, a_min=qmin, a_max=qmax).astype(
        nptype_from_name("uint", precision)
    )
    return data, scale, zero_point


def quantize_post_training_alphas(estimator, precision, min_val=None, max_val=None):
    assert precision in [8, 16, 32], "Precision must be 8, 16 or 32"
    if min_val is None:
        logging.warning("No min_val provided, computing min_val from alphas")
        min_val = min(
            [
                node.alpha
                for tree in estimator.flat_trees
                for node in preorder_iter(tree)
                if not node.is_leaf
            ]
        )
    if max_val is None:
        logging.warning("No max_val provided, computing max_val from alphas")
        max_val = max(
            [
                node.alpha
                for tree in estimator.flat_trees
                for node in preorder_iter(tree)
                if not node.is_leaf
            ]
        )
    assert min_val < max_val, "min_val must be less than max_val"

    qestimator = deepcopy(estimator)

    # Quantize alphas, iterate over trees even in OVO case (flat list)
    for tree in qestimator.flat_trees:
        for node in preorder_iter(tree):
            if not node.is_leaf:
                node.alpha, scale, zero_point = quantize(
                    data=node.alpha,
                    min_val=min_val,
                    max_val=max_val,
                    precision=precision,
                )

    qestimator.alpha_scale = scale
    qestimator.alpha_zero_point = zero_point

    return qestimator


def quantize_pre_training_alphas(estimator, precision, min_val, max_val):
    assert precision in [8, 16, 32], "Precision must be 8, 16 or 32"

    # Unsigned quantization
    scale = (max_val - min_val) / (2**precision - 1)
    zero_point = -(round(min_val / scale) - 0)

    qestimator = deepcopy(estimator)
    qestimator.alpha_scale = scale
    qestimator.alpha_zero_point = zero_point
    # Quantize alphas, iterate over trees even in OVO case (flat list)
    for tree in qestimator.flat_trees:
        for node in preorder_iter(tree):
            if not node.is_leaf:
                node.alpha = np.ceil(node.alpha).astype(
                    nptype_from_name("uint", precision)
                )
    return qestimator


def quantize_leaves(estimator, precision, aggreagate="sum"):
    """
    Quantize leaves of a tree ensemble.

    Parameters
    ----------
    estimator : _type_
        _description_
    precision : _type_
        _description_
    aggreagate : str, optional
        _description_, by default "sum"
    """
    assert aggreagate in [
        "sum"
    ], "Aggregate must be sum, other approaches are not supported yet"
    assert precision in [8, 16, 32], "Precision must be 8, 16 or 32"

    qestimator = deepcopy(estimator)
    min_val, max_val = qestimator.leaf_range
    # Quantize leaves
    for tree in qestimator.flat_trees:
        for leaf in tree.leaves:
            leaf.values, scale, zero_point = quantize(
                data=leaf.values,
                min_val=min_val,
                max_val=max_val,
                precision=precision,
                method="trunc",
            )
    
    qestimator.leaf_scale = scale
    qestimator.leaf_zero_point = zero_point

    return qestimator
