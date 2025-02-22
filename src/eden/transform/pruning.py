from eden.model.ensemble import Ensemble
import numpy as np
from eden.model.node import Node
from bigtree import preorder_iter, print_tree
from copy import deepcopy


def prune_same_class_leaves(*, estimator):
    """
    Prune leaves that have the same class in all of their samples.
    """
    assert estimator.task not in ["regression", "multiclass_classification_ovo"]
    pestimator = deepcopy(estimator)
    for idx, tree in enumerate(pestimator.trees):
        stop_flag = True
        while stop_flag:
            stop_flag = False
            for node in tree.leaves:
                if (
                    node.siblings[0].is_leaf
                    and node.values.argmax() == node.siblings[0].values.argmax()
                ):
                    parent = node.parent
                    zp = np.zeros_like(parent.values)
                    zp[:, node.values.argmax()] = 1
                    parent.values = zp
                    parent.left = None
                    parent.right = None
                    stop_flag = True
                    break
        # Cleanup
        for node in preorder_iter(tree):
            node.values = (
                np.argmax(node.values)
                .reshape(1, 1)
                .astype(np.min_scalar_type(node.values.shape))
            )

    # Change the estimator task
    pestimator.task = "classification_label"

    # Swap the leaves witht their argmax
    return pestimator


def prune_cost_complexity(*, tree, alpha):
    raise NotImplementedError("Use the training library to prune")
    # TODO: Implement this, however, can we do it with the validation set?
    # Iterate over each subtree of the tree

    # For each subtree, compute:
    # 1. R(T_t) -> The risk of the leaves: sum of the gini impurity of each leaf
    # 2. R(T) -> The risk of the node: gini * (n_samples_to_node / n_samples_total)
    # 3. |T_t| The number of leaves
    # 4. The effective cost: (R(t) - R(T_t))/ (|T_t| - 1)

    # Prune subtree if alpha larger than 4.
