from typing import Any, Optional, List, Tuple
import numpy as np
from copy import deepcopy

from sklearn import tree
from sklearn import ensemble
from sklearn import base


def _get_leaves_bounds(
    *, n_trees: int, task: str, n_estimators: int, leaf_shape: int, leaves: np.ndarray
):
    minima = np.zeros(shape=(n_trees, leaf_shape))
    maxima = np.zeros(shape=(n_trees, leaf_shape))

    for idx, tree_leaves in enumerate(leaves):
        t_min = np.min(tree_leaves, axis=0)
        t_max = np.max(tree_leaves, axis=0)
        minima[idx] = t_min
        maxima[idx] = t_max

    # GBT-like
    if task == "classification-ova":
        minima = minima.reshape((n_estimators, n_trees / n_estimators))
        maxima = maxima.reshape((n_estimators, n_trees / n_estimators))

    minima = np.sum(minima, axis=0)
    maxima = np.sum(maxima, axis=0)
    minimum = np.floor(np.min(minima))
    maximum = np.ceil(np.max(maxima))

    return minimum, maximum


def _parse_sklearn_tree(
    *, tree: Any, task: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_tree: tree.Tree = tree.tree_
    threshold = base_tree.threshold
    right_child = base_tree.children_right
    feature_idx = base_tree.feature
    # Right index to shift, 0 for leaf nodes
    for i in range(right_child.shape[0]):
        if right_child[i] < 0:
            right_child[i] = 0
        else:
            right_child[i] -= i
    # Difficult to obtain later, saved here
    leaf_nodes = feature_idx == -2
    leaf = base_tree.value[leaf_nodes]
    # Leaf processing
    leaf = np.squeeze(leaf)
    if task == "regression" or task == "regression-ova":
        leaf = leaf.reshape(-1, 1)
    # TODO: Understand how this changes depending on fitting weights
    if task == "classification":
        leaf = leaf / base_tree.weighted_n_node_samples[leaf_nodes, None]
        if tree.n_classes_ == 2:
            leaf = leaf[:, 1].reshape(leaf.shape[0], 1)
    return (feature_idx, threshold, right_child, leaf)


def _parse_sklearn_model(
    *,
    model: Any,
) -> Tuple[
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    List[np.ndarray],
    str,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    model = deepcopy(model)
    task = "regression" if isinstance(model, base.RegressorMixin) else "classification"
    if isinstance(model, tree.DecisionTreeClassifier) or isinstance(
        model, tree.DecisionTreeRegressor
    ):
        n_estimators = 1
        base_trees = [model]
    elif isinstance(model, ensemble._forest.BaseForest):
        base_trees = model.estimators_
        n_estimators = model.n_estimators
    elif isinstance(model, ensemble._gb.BaseGradientBoosting):
        n_estimators = model.n_estimators
        base_trees = model.estimators_.reshape(-1)
        if task == "classification":
            task = "classification-ova"
    else:
        raise ValueError(f"model={type(model)} is unsupported")

    # Ensemble struct
    right_children: List(np.ndarray) = list()
    thresholds: List(np.ndarray) = list()
    feature_idx: List(np.ndarray) = list()
    leaves: List[np.ndarray] = list()
    roots: List[int] = []
    starting_node: int = 0
    for albero in base_trees:
        f_idx, threshold, right_child, leaf = _parse_sklearn_tree(
            tree=albero,
            task=task,
        )
        roots.append(starting_node)
        feature_idx.append(f_idx)
        thresholds.append(threshold)
        right_children.append(right_child)
        leaves.append(leaf)
        starting_node += len(threshold)

    # Ensemble stats
    n_features = model.n_features_in_
    max_depth = max([t.tree_.max_depth for t in base_trees])
    n_trees = len(feature_idx)
    n_leaves = sum([l.shape[0] for l in leaves])
    n_nodes = sum([n.shape[0] for n in thresholds])
    leaf_shape = leaves[-1].shape[1]
    output_shape = model.n_outputs_
    roots = np.asarray(roots)

    return (
        roots,
        feature_idx,
        thresholds,
        right_children,
        leaves,
        task,
        n_features,
        max_depth,
        n_estimators,
        n_trees,
        n_leaves,
        n_nodes,
        leaf_shape,
        output_shape,
    )
