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

"""
Collections of functions to determine the L-Flag for each variable,
it requires an estimation of the memory
"""
import math
from typing import Mapping
from copy import deepcopy
import numpy as np


def _knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    selected = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                included_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                excluded_value = dp[i - 1][w]

                if included_value > excluded_value:
                    dp[i][w] = included_value
                    selected[i][w] = True
                else:
                    dp[i][w] = excluded_value

    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if selected[i][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items


# For now it is used only by GAP8
# Set in L1 everything fitting, leave the rest in L2 (no flag for GAP8)
# TODO : Last function to be called, this requires an idea of external/internal leaves,
def cache_placer(
    estimator_dict: Mapping[str, int],
    input_bits: int,
    output_bits: int,
)->Mapping:
    """
    Determines the cache level of each C array of the ensemble

    Parameters
    ----------
    estimator_dict : Mapping[str, int]
        The eden ensemble dictionary
    input_bits, output_bits : int
        Bit width of inputs/outputs

    Returns
    -------
    Mapping
        The updated estimator dict
    """
    # The order depends on the access ratio.
    estimator_dict = deepcopy(estimator_dict)
    cache = {}
    cache["EDEN_NODE_ARRAY"] = {}
    cache["EDEN_NODE_STRUCT"] = {}
    estimator_dict["input_bits"] = input_bits
    estimator_dict["threshold_bits"] = input_bits
    estimator_dict["output_bits"] = output_bits
    input_len = estimator_dict["input_len"]
    input_bits = estimator_dict["input_bits"]
    n_nodes = estimator_dict["n_nodes"]
    output_len = estimator_dict["output_len"]
    output_bits = estimator_dict["output_bits"]
    n_estimators = estimator_dict["n_estimators"]
    n_trees = estimator_dict["n_trees"]
    feature_bits = estimator_dict["feature_bits"]
    threshold_bits = estimator_dict["threshold_bits"]
    right_child_shift_bits = estimator_dict["children_right_bits"]

    leaves_bits = estimator_dict["output_bits"]
    n_leaves = estimator_dict["n_leaves"]
    SIZE_L1 = 64000

    # Inputs
    input_buffer_weight = input_len * input_bits
    input_buffer_value = n_nodes

    # Output
    output_buffer_weight = output_len * output_bits
    output_buffer_value = n_estimators * output_len

    # Roots
    roots_len = n_trees
    roots_buffer_weight = estimator_dict["roots_bits"] * n_trees
    roots_buffer_value = n_trees

    # Nodes
    # TODO : Include alignment
    nodes_buffer_weight = (
        feature_bits + threshold_bits + right_child_shift_bits
    ) * n_nodes
    nodes_buffer_value = math.log2(n_nodes)
    # Knapsack
    weights = [
        input_buffer_weight,
        output_buffer_weight,
        roots_buffer_weight,
        nodes_buffer_weight,
    ]
    values = [
        input_buffer_value,
        output_buffer_value,
        roots_buffer_value,
        nodes_buffer_value,
    ]
    # Leaves
    if estimator_dict["leaves_store_mode"] == "external":
        leaves_buffer_weight = (output_bits) * n_leaves
        leaves_buffer_value = math.log2(n_leaves)
        weights.append(leaves_buffer_weight)
        values.append(leaves_buffer_value)

    max_val, idx_in_sack = _knapsack_01(
        weights=weights, values=values, capacity=SIZE_L1
    )
    cache["EDEN_NODE_STRUCT"]["input_ltype"] = "L1" if 0 in idx_in_sack else ""
    cache["EDEN_NODE_STRUCT"]["output_ltype"] = "L1" if 1 in idx_in_sack else ""
    cache["EDEN_NODE_STRUCT"]["roots_ltype"] = "L1" if 2 in idx_in_sack else ""
    cache["EDEN_NODE_STRUCT"]["nodes_ltype"] = "L1" if 3 in idx_in_sack else ""

    if estimator_dict["leaves_store_mode"] == "external":
        cache["EDEN_NODE_STRUCT"]["leaves_ltype"] = "L1" if 4 in idx_in_sack else ""

    # For array-mode
    # Feature
    feature_buffer_weight = (feature_bits) * n_nodes
    feature_buffer_value = math.log2(n_nodes)
    # Threshold
    threshold_buffer_weight = (threshold_bits) * n_nodes
    threshold_buffer_value = math.log2(n_nodes)

    # Right child
    right_child_shift_buffer_weight = (right_child_shift_bits) * n_nodes
    right_child_shift_buffer_value = math.log2(n_nodes)
    weights = [
        input_buffer_weight,
        output_buffer_weight,
        roots_buffer_weight,
        feature_buffer_weight,
        threshold_buffer_weight,
        right_child_shift_buffer_weight,
    ]
    values = [
        input_buffer_value,
        output_buffer_value,
        roots_buffer_value,
        feature_buffer_value,
        threshold_buffer_value,
        right_child_shift_buffer_value,
    ]
    # Leaves
    if estimator_dict["leaves_store_mode"] == "external":
        weights.append(leaves_buffer_weight)
        values.append(leaves_buffer_value)

    _, idx_in_sack = _knapsack_01(weights=weights, values=values, capacity=SIZE_L1)
    cache["EDEN_NODE_ARRAY"]["input_ltype"] = "L1" if 0 in idx_in_sack else ""
    cache["EDEN_NODE_ARRAY"]["output_ltype"] = "L1" if 1 in idx_in_sack else ""
    cache["EDEN_NODE_ARRAY"]["roots_ltype"] = "L1" if 2 in idx_in_sack else ""
    cache["EDEN_NODE_ARRAY"]["feature_ltype"] = "L1" if 3 in idx_in_sack else ""
    cache["EDEN_NODE_ARRAY"]["threshold_ltype"] = "L1" if 4 in idx_in_sack else ""
    cache["EDEN_NODE_ARRAY"]["children_right_ltype"] = "L1" if 5 in idx_in_sack else ""

    if estimator_dict["leaves_store_mode"] == "external":
        cache["EDEN_NODE_ARRAY"]["leaves_ltype"] = "L1" if 6 in idx_in_sack else ""

    estimator_dict["cache"] = cache
    return estimator_dict
