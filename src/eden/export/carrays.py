from bigtree import BinaryNode, print_tree, preorder_iter
from eden.utils import _compute_nptype
import numpy as np


def ensemble_to_c_arrays(
    ensemble,
):
    children_right, features, alphas, leaves = [], [], [], []
    roots = []
    n_leaves = 0
    n_nodes = 0
    for tree in ensemble.flat_trees:
        cr, f, al, le = tree_to_c_arrays(tree=tree)
        # Leaves outside
        leaves_idx = f == (ensemble.input_length)
        if ensemble.task == "classification_multiclass" and le.shape[-1] > 2:
            # Update the dtype in case
            leaf_range = np.arange(n_leaves, n_leaves + le.shape[0])
            cr = cr.astype(max(np.min_scalar_type(leaf_range.max()), cr.dtype))
            cr[leaves_idx] = leaf_range
            leaves.append(le)
        # Leaves inside, directly
        elif ensemble.task in ["classification_multiclass_ovo", "regression"]:
            le_typed = le.reshape(-1).astype(np.min_scalar_type(le.max()))
            al = al.astype(max(le_typed.dtype, al.dtype))
            al[leaves_idx] = le.reshape(-1)
        # Leaves inside, but in the right
        elif ensemble.task == "classification_label":
            # Argmax has already been performed before
            classes = le
            cr = cr.astype(max(np.min_scalar_type(ensemble.n_trees), cr.dtype))
            cr[leaves_idx] = classes.reshape(-1)
        # Leaves inside, binary classification case
        else:
            le_typed = le[:, 1].reshape(-1).astype(np.min_scalar_type(le.max()))
            al = al.astype(max(le_typed.dtype, al.dtype))
            al[leaves_idx] = le_typed

        roots.append(n_nodes)
        children_right.append(cr)
        features.append(f)
        alphas.append(al)
        # Update the number of leaves and nodes
        n_leaves += le.shape[0]
        n_nodes += len(al)

    # Create the arrays
    children_right = np.concatenate(children_right)
    roots = np.array(roots)
    alphas = np.concatenate(alphas)
    if leaves != list():
        leaves = np.concatenate(leaves)
    else:
        leaves = None
    features = np.concatenate(features)

    # Cast to correct dtype
    features = features.astype(np.min_scalar_type(ensemble.input_length))
    children_right = children_right.astype(np.min_scalar_type(children_right.max()))
    roots = roots.astype(np.min_scalar_type(roots.max()))

    # Never change this order, half package depends on it
    return children_right, features, alphas, leaves, roots


def tree_to_c_arrays(tree):
    # Inefficient, yet easy to program, a better way would be to make it recursive
    assert tree.is_root, "Export starts only from the root node"
    input_length = tree.input_length
    preorder_nodes = list(preorder_iter(tree))
    alphas, features, children_right, values = (
        np.zeros(len(preorder_nodes), dtype=tree.alpha.dtype),
        np.zeros(len(preorder_nodes), dtype=np.min_scalar_type(tree.input_length)),
        np.zeros(len(preorder_nodes), dtype=np.min_scalar_type(2**tree.max_depth)),
        np.zeros(
            (len(preorder_nodes), tree.values.shape[-1]),
            dtype=next(tree.leaves).values.dtype,
        ),
    )
    for node_idx, node in enumerate(preorder_nodes):
        # Shift is the number of nodes between the current node and the right child
        # in the preorder traversal
        if node.right is not None:
            children_right[node_idx] = preorder_nodes.index(node.right) - node_idx
        else:
            # Set shift to 0 for leaves
            children_right[node_idx] = 0

        if node.is_leaf:
            features[node_idx] = input_length
        else:
            features[node_idx] = node.feature

        alphas[node_idx] = node.alpha
        values[node_idx] = node.values

    # Get only the leaves
    leaves = values[features == (input_length), :]

    # DTypes of indexes are still int32, they are converted in the ensemble class
    return children_right, features, alphas, leaves
