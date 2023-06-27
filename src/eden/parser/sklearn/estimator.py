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

import numpy as np
from typing import Union, MutableMapping, Mapping, List, Iterable
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._forest import BaseForest
from sklearn import base
import eden
from .tree import parse_tree_data, parse_tree_info
from sklearn.tree import BaseDecisionTree
from sklearn.base import ClassifierMixin


def parse_estimator_info(*, estimator):
    input_len = estimator.n_features_in_
    if hasattr(estimator, "estimators_"):
        n_estimators = estimator.n_estimators
        n_trees = len(estimator.estimators_)
    else:
        n_estimators = 1
        n_trees = 1

    output_len = estimator.n_classes_ if isinstance(estimator, ClassifierMixin) else 1
    estimators = (
        [estimator]
        if not hasattr(estimator, "estimators_")
        else np.asarray(estimator.estimators_).reshape(-1)
    )

    n_nodes = 0
    n_leaves = 0
    max_depth = -1
    for tree in estimators:
        max_depth_tree, n_nodes_tree, n_leaves_tree, leaf_len = parse_tree_info(
            estimator=tree
        )
        n_nodes += n_nodes_tree
        n_leaves += n_leaves_tree
        if max_depth_tree > max_depth:
            max_depth = max_depth_tree

    return (
        n_estimators,
        n_trees,
        max_depth,
        input_len,
        n_nodes,
        n_leaves,
        leaf_len,
        output_len,
    )


def parse_estimator_data(*, estimator):
    root, feature, threshold, children_right, children_left, leaf_value = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    estimators = (
        [estimator]
        if not hasattr(estimator, "estimators_")
        else np.asarray(estimator.estimators_).reshape(-1)
    )
    # A single tree
    n_nodes = 0
    # An ensemble
    for tree in estimators:
        root.append(n_nodes)
        (
            tree_feature,
            tree_threshold,
            tree_children_right,
            tree_children_left,
            tree_leaf_value,
        ) = parse_tree_data(estimator=tree)
        feature.append(tree_feature)
        threshold.append(tree_threshold)
        children_right.append(tree_children_right)
        children_left.append(tree_children_left)
        leaf_value.append(tree_leaf_value)
        n_nodes += len(tree_threshold)

    root = np.asarray(root)
    feature = np.concatenate(feature)
    threshold = np.concatenate(threshold)
    children_right = np.concatenate(children_right)
    children_left = np.concatenate(children_left)
    leaf_value = np.concatenate(leaf_value)

    return root, feature, threshold, children_right, children_left, leaf_value
