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

from typing import Union, Mapping, Any, Tuple, List
from sklearn.tree._tree import TREE_LEAF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
import numpy as np
import eden
from copy import deepcopy


def _visit_tree(tree):
    lc = tree.children_left
    rc = tree.children_right
    stack = list()
    pre_order = list()
    stack.append(0)  # Root
    while len(stack) > 0:
        idx_nodo = stack.pop()
        if idx_nodo >= 0:
            pre_order.append(idx_nodo)
            stack.append(rc[idx_nodo])
            stack.append(lc[idx_nodo])
    return np.asarray(pre_order)


def _get_eden_leaf_value(n_features: int):
    # Used to keep track of all functions where it is used
    return n_features + 1


def parse_tree_data(
    *,
    estimator: Union[DecisionTreeRegressor, DecisionTreeClassifier],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sklearn-specific tree extraction, handles all the nodes arrays

    Parameters
    ----------
    estimator : Union[DecisionTreeRegressor, DecisionTreeClassifier]
        The tree object to be parsed

    Notes
    ----------
    This is the low level implementation, returning the arrays in a numpy format
    for further processing. Leaves have feature index equal to tree_.LEAF
    """
    base_tree = estimator.tree_
    # All the values read need to be changed with this logic
    pre_order = _visit_tree(base_tree)
    n_nodes = len(pre_order)  # Changed to account pruning

    # Indexes of leaves in the arrays
    leaves_idxs = base_tree.children_left[pre_order] == _tree.TREE_LEAF
    nodes_idxs = ~leaves_idxs
    # Extract the arrays - ALWAYS with pre-order, as there may be pruned indexes
    children_left = base_tree.children_left[
        pre_order
    ]  # Not used in C, but maybe in future?
    # Before taking them in pre-order and post-pruning, we swap the index representation
    # to the shift-based one
    # Right index to shift, 0 for leaf nodes
    children_right = np.copy(base_tree.children_right)
    for i in range(len(base_tree.children_right)):
        if children_right[i] != TREE_LEAF:
            children_right[i] = children_right[i] - i
        else:
            children_right[i] = 0
    assert np.max((children_right[pre_order] + np.arange(len(pre_order)))) < len(
        pre_order
    )
    children_right = base_tree.children_right[pre_order]
    threshold = base_tree.threshold[pre_order]
    feature = base_tree.feature[pre_order]


    # Save all nodes values for future works, only leaves are stored in C
    node_value = base_tree.value[pre_order]  # [N_NODES, 1, N_CLASSES]
    # Remove the unused middle dimension
    node_value = np.squeeze(node_value).reshape(
        len(threshold), -1
    )  # [N_NODES, N_CLASSES]

    # Classification tree only (excludes GBT for classification)
    if isinstance(estimator, DecisionTreeClassifier):
        # Save only the probabilities, not the sample counts
        node_value = node_value / node_value.sum(-1)[:, None]
        # In case of 2 classes, remove the class 0
        if estimator.n_classes_ == 2:
            # Reshape is needed to ensure [N_NODES, N_CLASSES]
            node_value = node_value[:, 1].reshape(node_value.shape[0], 1)

    # REMAP - LEAF VALUE -> EDEN CUSTOM
    children_right[leaves_idxs] = 0
    threshold[leaves_idxs] = 0
    feature[leaves_idxs] = _get_eden_leaf_value(n_features=base_tree.n_features)
    leaf_value = node_value[leaves_idxs]
    assert (children_right == 0).sum() == len(leaf_value)

    return (
        feature,  # [N_NODES]
        threshold,  # [N_NODES]
        children_right,  # [N_NODES]
        children_left,  # [N_NODES]
        leaf_value,  # [N_NODES, N_CLASSES]
    )


def parse_tree_info(*, estimator):
    base_tree = estimator.tree_
    pre_order = _visit_tree(base_tree)
    max_depth = base_tree.max_depth  # TODO: Fix for pruning
    # We cannot use the base values if we pruned
    n_nodes = len(base_tree.value[pre_order])
    n_leaves = (base_tree.children_left[pre_order] == _tree.TREE_LEAF).sum()
    leaf_len = base_tree.value.shape[-1]
    return max_depth, n_nodes, n_leaves, leaf_len
