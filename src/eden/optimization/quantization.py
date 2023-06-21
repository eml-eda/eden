# *--------------------------------------------------------------------------*
# * Copyright (c) 2023 Politecnico di Torino, Italy                          *
# * SPDX-License-Identifier: Apache-2.0                                      *
# *                                                                          *
# * Licensed under the Apache License, Version 2.0 (the "License");          *
# * you may not use this file except in compliance with the License.         *
# * You may obtain a copy of the License at                                  *
# *                                                                          *
# * http://www.apache.org/licenses/LICENSE-2.0                               *
# *                                                                          *
# * Unless required by applicable law or agreed to in writing, software      *
# * distributed under the License is distributed on an "AS IS" BASIS,        *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
# * See the License for the specific language governing permissions and      *
# * limitations under the License.                                           *
# *                                                                          *
# * Author: Francesco Daghero francesco.daghero@polito.it                    *
# *--------------------------------------------------------------------------*

from sklearn.base import BaseEstimator
from typing import Literal, Tuple
import numpy as np
from copy import deepcopy
from sklearn.tree import _tree, DecisionTreeClassifier, BaseDecisionTree


def get_qparams(*, min_val: float, max_val: float, bits: Literal[8, 16, 32]):
    """
    Get a dictionary with scale and zero point selected

    Parameters
    ----------
    data : np.ndarray
        Data to be quantized
    min_val , max_val : float
        Range of the data to be quantized
    bits : Literal[8,16,32]
        QData bit width.
    method : str
        Function to apply after scale and zero point, default is clip

    Returns
    -------
    Dict
        Qparams of the quantization
    """
    qmin = 0
    qmax = 2 ** (bits) - 1
    S = (max_val - min_val) / (2**bits - 1)
    Z = -(round(min_val / S) - qmin)
    return {"scale": S, "zero_point": Z}


def quantize(
    *,
    data: np.ndarray,
    min_val: float,
    max_val: float,
    bits: Literal[8, 16, 32] = 32,
    method: str = "clip",
) -> np.ndarray:
    """
    Base quantization function for thresholds and inputs.

    Parameters
    ----------
    data : np.ndarray
        Data to be quantized
    min_val , max_val : float
        Range of the data to be quantized
    bits : Literal[8,16,32]
        QData bit width.
    method : str
        Function to apply after scale and zero point, default is clip

    Returns
    -------
    np.ndarray
        A quantized copy of data
    """
    data = np.copy(data)
    qmin = 0
    qmax = 2 ** (bits) - 1
    S = (max_val - min_val) / (2**bits - 1)
    Z = -(round(min_val / S) - qmin)
    if method == "clip":
        data = np.round(data / S + Z).astype(int)
    else:
        data = np.trunc(data / S + Z).astype(int)
    data = np.clip(a=data, a_min=qmin, a_max=qmax)
    return data


def quantize_alphas(
    *,
    estimator: BaseEstimator,
    input_min_val: float,
    input_max_val: float,
    method: Literal["post-fit", "qat"] = "post-fit",
    bits: Literal[8, 16, 32] = 8,
) -> BaseEstimator:
    """
    Quantize the thresholds (alphas) in the input estimator.

    Parameters
    ----------
    estimator : BaseEstimator
        The estimator to be modified
    input_min_val, input_max_val : float
        Range of the input data.
    method : str, optional
        Quantization method, "post-fit" for post-training, "qat" for quant-aware
          training, by default "post-fit"
    bits : Literal[8, 16, 32], optional
        Quantization bits of the thresholds, by default 8

    Returns
    -------
    BaseEstimator
        A copy of estimator with quantized alphas.
    """
    assert hasattr(estimator, "estimators_") or hasattr(
        "tree_"
    ), "Invalid or not trained model, call .fit() first"
    qestimator = deepcopy(estimator)
    # Ensemble - GBT
    if hasattr(estimator, "tree_") and not hasattr(estimator, "estimators_"):
        __quantize_tree_alphas(
            qestimator,
            min_val=input_min_val,
            max_val=input_max_val,
            bits=bits,
            method=method,
        )
    else:
        for tree in np.asarray(estimator.estimators_).reshape(-1):
            __quantize_tree_alphas(
                tree,
                min_val=input_min_val,
                max_val=input_max_val,
                bits=bits,
                method=method,
            )
    return qestimator


def __quantize_tree_alphas(
    *, tree: BaseDecisionTree, min_val: float, max_val: float, bits: int, method: str
):
    basetree = tree.tree_
    leaves_idx = basetree.children_left != _tree.TREE_LEAF
    if method == "qat":
        basetree.threshold[leaves_idx] = np.ceil(basetree.threshold[leaves_idx])
    else:
        basetree.threshold[leaves_idx] = quantize(
            basetree.threshold[leaves_idx], min_val=min_val, max_val=max_val, bits=bits
        )


#######################################################################################
## Output quantization


def __extract_tree_leaves(estimator):
    base_tree = deepcopy(estimator.tree_)

    # Remove the unused middle dimension
    leaf_value = np.squeeze(base_tree.value).reshape(
        len(base_tree.threshold), -1
    )  # [N_NODES, N_CLASSES]
    leaf_value = leaf_value[
        base_tree.children_left == _tree.TREE_LEAF
    ]  # [N_LEAVES, 1, N_CLASSES]

    # Classification tree only (excludes GBT for classification)
    if isinstance(estimator, DecisionTreeClassifier):
        # Save only the probabilities, not the sample counts
        leaf_value = leaf_value / leaf_value.sum(axis=1)[:, None]
        if estimator.n_classes_ == 2:
            # Reshape is needed to ensure [N_NODES, N_CLASSES]
            leaf_value = leaf_value[:, 1].reshape(leaf_value.shape[0], 1)
    return leaf_value


def __get_leaf_extrema(estimator) -> Tuple[float, float]:
    if hasattr(estimator, "tree_") and not hasattr(estimator, "estimators_"):
        leaf_shape = (
            estimator.tree_.value.shape[-1]
            if not hasattr(estimator.tree_, "n_classes_")
            else 1
        )
        extrema = np.zeros((1, leaf_shape, 2))
        leaf_value = __extract_tree_leaves(estimator=estimator)

        extrema[0, :, 0] = np.min(leaf_value, axis=0)
        extrema[0, :, 1] = np.max(leaf_value, axis=0)
    else:
        leaf_shape = estimator.estimators_[0].tree_.value.shape[-1]
        extrema = np.zeros(
            (len(np.asarray(estimator.estimators_).reshape(-1)), leaf_shape, 2)
        )
        for idx, tree in enumerate(np.asarray(estimator.estimators_).reshape(-1)):
            leaf_value = __extract_tree_leaves(estimator=tree)

            extrema[idx, :, 0] = np.min(leaf_value, axis=0)
            extrema[idx, :, 1] = np.max(leaf_value, axis=0)

        if hasattr(estimator, "classes_") and leaf_shape != estimator.n_classes_:
            extrema = extrema.reshape(
                len(estimator.estimators_), estimator.n_classes_, 2
            )
    if extrema.shape[1] == 2:
        extrema = extrema[:, 1, :]
    if hasattr(estimator, "init") and estimator.init != "zero":
        extrema += estimator.init_[None, :, None]

    extrema = extrema.sum(axis=0)
    output_min, output_max = extrema.min(), extrema.max()
    return output_min, output_max


def get_output_qparams(estimator):
    assert hasattr(estimator, "estimators_") or hasattr(
        "tree_"
    ), "Invalid or not trained model, call .fit() first"
    qestimator = deepcopy(estimator)
    output_min, output_max = __get_leaf_extrema(qestimator)
    return output_min, output_max
