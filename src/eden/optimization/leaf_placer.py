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
Collection of functions to detect where the leaf data is stored.
Generally internal for binary or regression, always external for multiclass
"""

from typing import Mapping, Optional
import numpy as np
from copy import deepcopy

from . import _get_bits_to_represent


def prepare_leaves_placement(
    *,
    estimator_dict: Mapping,
    leaf_placement_strategy: str,
    input_qbits: Optional[int],
    output_qbits: Optional[int],
)->Mapping:
    """
    Checks if leaves should be stored inside or outside the nodes.

    Parameters
    ----------
    estimator_dict : Mapping
        The eden dictionary
    leaf_placement_strategy : str
        How the leaves should be stored, possible values : internal, external, auto.
    input_qbits, output_qbits : Optional[int]
        input and output bitwidth, None means float

    Returns
    -------
    Mapping
        The updated eden dictionary
    """
    estimator_dict = deepcopy(estimator_dict)
    n_nodes = estimator_dict["n_nodes"]
    n_leaves = estimator_dict["n_leaves"]
    leaf_len = estimator_dict["leaf_len"]
    if input_qbits is None:
        input_qbits = 32
    if output_qbits is None:
        output_qbits = 32

    max_shift = max([max(t["children_right"]) for t in estimator_dict["trees"]])
    right_child_shift_bits = _get_bits_to_represent(range_val=(0, max_shift))
    leaves_idx_bits = _get_bits_to_represent(range_val=(0, n_leaves))

    # Compute the memory required to store the leaves outside
    external_memory = n_nodes * input_qbits + n_leaves * output_qbits
    # Right child may become larger
    external_memory += n_nodes * max(right_child_shift_bits, leaves_idx_bits)

    internal_memory = n_nodes * max(input_qbits, output_qbits)
    internal_memory += n_nodes * right_child_shift_bits

    if leaf_len != 1:
        leaf_placement_strategy = "external"
    else:
        if leaf_placement_strategy == "auto":
            leaf_placement_strategy = (
                "internal" if internal_memory < external_memory else "external"
            )
        else:
            leaf_placement_strategy = "external"

    # No actions yet on the values to avoid messing the quantization.
    # Right child can be changed
    if leaf_placement_strategy=="external":
        previous_leaves = 0
        for idx, tree in enumerate(estimator_dict["trees"]):
            leaves_idxs = np.asarray(tree["feature"]) == tree["eden_leaf_indicator"]
            right_child = np.asarray(tree["children_right"])
            right_child[leaves_idxs] = np.arange(0, tree["n_leaves"])
            right_child[leaves_idxs] += previous_leaves
            # Re-write in the dictionary
            tree["children_right"] = right_child.tolist()
            previous_leaves += tree["n_leaves"]
    return estimator_dict, leaf_placement_strategy


def merge_leaves_in_thresholds(estimator_dict : Mapping):
    estimator_dict = deepcopy(estimator_dict)
    for tree in estimator_dict["trees"]:
        leaves_idxs = np.asarray(tree["feature"]) == tree["eden_leaf_indicator"]
        threshold = np.asarray(tree["threshold"])
        threshold[leaves_idxs] = np.asarray(tree["values"])
        tree["threshold"]= threshold.tolist()
    return estimator_dict
