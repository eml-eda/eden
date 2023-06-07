from typing import Union, Mapping, Any, Tuple, List
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, _tree
import numpy as np
import eden
from copy import deepcopy


def _get_eden_leaf_value(n_features: int):
    # Used to keep track of all functions where it is used
    return n_features + 1


def _parse_tree(
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
    n_nodes = len(base_tree.children_right)
    # Indexes of leaves in the arrays
    leaves_idxs = base_tree.children_left == _tree.TREE_LEAF
    nodes_idxs = ~leaves_idxs
    # Extract the arrays
    children_left = base_tree.children_left  # Not used in C, but maybe in future?
    children_right = base_tree.children_right
    threshold = base_tree.threshold
    feature_idx = base_tree.feature
    # Right index to shift, 0 for leaf nodes
    children_right[nodes_idxs] -= np.arange(n_nodes)[nodes_idxs]
    # Save all nodes values for future works, only leaves are stored in C
    node_value = base_tree.value  # [N_NODES, 1, N_CLASSES]
    # Remove the unused middle dimension
    node_value = np.squeeze(node_value).reshape(
        len(threshold), -1
    )  # [N_NODES, N_CLASSES]

    # Classification tree only (excludes GBT for classification)
    if isinstance(estimator, DecisionTreeClassifier):
        # Save only the probabilities, not the sample counts
        node_value = node_value / base_tree.weighted_n_node_samples[:, None]
        # In case of 2 classes, remove the class 0
        if estimator.n_classes_ == 2:
            # Reshape is needed to ensure [N_NODES, N_CLASSES]
            node_value = node_value[:, 1].reshape(node_value.shape[0], 1)

    # REMAP - LEAF VALUE -> EDEN CUSTOM
    children_right[leaves_idxs] = 0
    threshold[leaves_idxs] = 0
    feature_idx[leaves_idxs] = _get_eden_leaf_value(n_features=base_tree.n_features)

    return (
        feature_idx,  # [N_NODES]
        threshold,  # [N_NODES]
        children_right,  # [N_NODES]
        children_left,  # [N_NODES]
        node_value,  # [N_NODES, N_CLASSES]
    )


def parse_tree(
    *,
    estimator: Union[DecisionTreeRegressor, DecisionTreeClassifier],
) -> Mapping[Any, Any]:
    """
    Converts a DecisionTree from sklearn in a serializable dictionary.

    Parameters
    ----------
    estimator : Union[DecisionTreeRegressor, DecisionTreeClassifier]
        The tree object to be converted

    Returns
    -------
    Mapping[Any, Any]
        Serializable dictionary following a specific JSON schema.
    """
    # Init the dictionary
    tree_dictionary: Mapping = {}
    # Get the numpy values
    feature_idx, threshold, children_right, children_left, node_values = _parse_tree(
        estimator=deepcopy(estimator)
    )

    # General info
    tree_dictionary["version"] = eden.__version__
    tree_dictionary["estimator"] = type(estimator).__name__
    # Tree info
    tree_dictionary["is_classification"] = isinstance(estimator, DecisionTreeClassifier)
    # Int cast is necessary, some fields are np.int64, not serializable
    tree_dictionary["n_nodes"] = int(estimator.tree_.node_count)
    tree_dictionary["n_leaves"] = int(estimator.tree_.n_leaves)
    tree_dictionary["max_depth"] = int(estimator.tree_.max_depth)
    tree_dictionary["leaf_len"] = (
        int(estimator.n_classes_) if hasattr(estimator, "n_classes_") else 1
    )
    tree_dictionary["input_len"] = int(estimator.n_features_in_)
    tree_dictionary["eden_leaf_indicator"] = _get_eden_leaf_value(
        n_features=int(estimator.n_features_in_)
    )

    # Serialize np.ndarray
    tree_dictionary["feature"] = feature_idx.tolist()
    tree_dictionary["threshold"] = threshold.tolist()
    tree_dictionary["children_right"] = children_right.tolist()
    tree_dictionary["children_left"] = children_left.tolist()
    tree_dictionary["value"] = node_values.tolist()

    return tree_dictionary
