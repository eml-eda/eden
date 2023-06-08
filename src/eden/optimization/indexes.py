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
Collection of scripts to compute the bits required by each index in the ensemble.
This optimization pass has no effect on the accuracy.
"""
from . import _get_bits_to_represent
from typing import Mapping, MutableMapping
from copy import deepcopy


def compute_bits_indexes(estimator_dict: Mapping) -> Mapping:
    """
    Determines the smallest bit-width for each index-array

    Parameters
    ----------
    estimator_dict : Mapping
        The eden dictionary

    Returns
    -------
    Mapping
        The updated eden dictionary
    """

    # Extract constants
    estimator_dict = deepcopy(estimator_dict)
    cfg = estimator_dict
    n_nodes = estimator_dict["n_nodes"]
    n_leaves = estimator_dict["n_leaves"]
    n_features = estimator_dict["input_len"]
    n_trees = estimator_dict["n_trees"]
    leaf_indicator = estimator_dict["eden_leaf_indicator"]
    # Derive bits for indexes
    # Derive indexes not impacting on accuracy
    cfg["roots_bits"] = _get_bits_to_represent(range_val=(0, n_nodes))
    # Node structure
    feature_range = [*range(0, n_features)]
    feature_range.append(leaf_indicator)
    cfg["feature_bits"] = _get_bits_to_represent(
        range_val=(min(feature_range), max(feature_range))
    )
    # Used only for external leaves
    cfg["leaf_idx_bits"] = _get_bits_to_represent(range_val=(0, n_leaves))

    # The upper bound is 2**D, but is generally much lower,
    # We extract it from the data
    max_shift = max([max(t["children_right"]) for t in estimator_dict["trees"]])
    cfg["children_right_bits"] = _get_bits_to_represent(range_val=(0, max_shift))
    return estimator_dict
