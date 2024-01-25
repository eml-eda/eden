from bigtree import BinaryNode, print_tree, preorder_iter
import numpy as np


def ensemble_to_arrays(*, ensemble):
    # Preorder array-based representation of the trees
    # Identical to sklearn
    children_left, children_right, features, alphas, values = [], [], [], [], []
    for tree in ensemble.flat_trees:
        cl, cr, f, thr, val = tree_to_arrays(tree)
        alphas.append(thr)
        features.append(f)
        children_right.append(cr)
        children_left.append(cl)
        values.append(val)
    return children_left, children_right, features, alphas, values


def tree_to_arrays(*, tree):
    # Inefficient, yet easy to program, a better way would be to make it recursive
    assert tree.is_root, "Export starts only from the root node"
    preorder_nodes = list(preorder_iter(tree))
    alphas, features, children_left, children_right, values = (
        np.zeros(len(preorder_nodes)),
        np.zeros(len(preorder_nodes), dtype=np.uint32),
        np.zeros(len(preorder_nodes), dtype=np.uint32),
        np.zeros(len(preorder_nodes), dtype=np.uint32),
        np.zeros((len(preorder_nodes), tree.values.shape[-1])),
    )
    for node_idx, node in enumerate(preorder_nodes):
        if node.right is not None:
            children_right[node_idx] = preorder_nodes.index(node.right)
        if node.left is not None:
            children_left[node_idx] = preorder_nodes.index(node.left)
        features[node_idx] = node.feature
        alphas[node_idx] = node.alpha
        values[node_idx] = node.values
    return children_left, children_right, features, alphas, values
