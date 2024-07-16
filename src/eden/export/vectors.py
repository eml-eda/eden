from bigtree import BinaryNode, print_tree, preorder_iter
from typing import List
from eden.utils import _compute_nptype
import numpy as np
from eden.model.ensemble import Ensemble
from eden.model.node import Node


def ensemble_to_vectors(ensemble: Ensemble, pad_to_perfect: bool = True):
    features, alphas, leaves, addr_maps, roots = [], [], [], [], []
    # Features: 2d array [N_VECTORS, C]
    # Alphas : 2d array [N_VECTORS, C]
    # Leaves : 2d array [N_VECTORS, LEAF_SHAPE]
    # addr_maps: 2d array [N_VECTORS, 2] -> [i, 0] == 1 only if (i) leads to a leaf,
    # roots: [N_TREES], stores the idx of the starting vector in Features, i.e. cumsum(n_leaves for each tree)

    max_depth = ensemble.max_depth - 1

    n_leaves = 0
    for idx, tree in enumerate(ensemble.flat_trees):
        fidx, al, le, addr = tree_to_vectors(
            tree, max_depth, pad_to_perfect=pad_to_perfect
        )
        # Update the addresses, scaling by the leaves and nodes
        addr[addr[:, 0] == 1, 1] += n_leaves
        addr[addr[:, 0] == 0, 1] += n_leaves

        features.append(fidx)
        alphas.append(al)
        leaves.append(le)
        addr_maps.append(addr)
        roots.append(n_leaves)

        n_leaves += le.shape[0]

    features = np.concatenate(features)
    alphas = np.concatenate(alphas)
    leaves = np.concatenate(leaves)
    addr_maps = np.concatenate(addr_maps)
    roots = np.asarray(roots)
    return roots, features, alphas, leaves, addr_maps


def tree_to_vectors(tree: Node, max_depth: int, pad_to_perfect=True):
    n_leaves = 2**max_depth if pad_to_perfect else tree.n_leaves
    features = np.zeros((n_leaves, max_depth), dtype=tree.feature.dtype)
    alphas = np.ones((n_leaves, max_depth), dtype=tree.alpha.dtype)
    leaves = np.zeros((n_leaves, tree.values.shape[-1]), dtype=tree.values.dtype)

    # Field 0: 1 is a leaf, 0 is a path to another branch
    # Field 1: Depending on field 0, an idx in leaves or in features/alphas
    addr_maps = np.ones((n_leaves, 2), dtype=np.uint32)

    # Since this is the function with C = max_depth ...
    addr_maps[:, 1] = np.arange(n_leaves)

    # Padding node -> A node that always returns true, since it is padding :P
    # Requires <= jumps to work
    if issubclass(alphas.dtype.type, np.integer):
        pad_val = np.iinfo(alphas.dtype).max
    else:
        pad_val = np.finfo(alphas.dtype).max
    alphas = alphas * pad_val

    for idx, leaf in enumerate(tree.leaves):
        path: List[Node] = tree.go_to(leaf)[:-1]
        for idx_node, node in enumerate(path):
            alphas[idx, idx_node] = node.alpha
            features[idx, idx_node] = node.feature
        leaves[idx] = leaf.values.reshape(-1)
    return features, alphas, leaves, addr_maps

