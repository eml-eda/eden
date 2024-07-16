from eden.model.ensemble import Ensemble
import numpy as np
from eden.model.node import Node
from bigtree import preorder_iter, print_tree
from copy import deepcopy
from bigtree import shift_nodes

def _get_max_value(dtype):
    if "int" in dtype.name:
        return dtype.type(np.iinfo(dtype).max)
    elif "float" in dtype.name:
        return dtype.type(np.finfo(dtype).max)


def _get_fake_threshold(value):
    val = _get_max_value(value.dtype)
    return val


def create_dummy_node(name: str, node: "Node", dummy: bool = False) -> "Node":
    # Return an identical node, but without parents/children
    is_dummy_node = dummy and not node.is_leaf 

    
    return Node(
        name=name,
        values=node.values,
        feature=node.feature,
        alpha = node.alpha if not is_dummy_node else _get_fake_threshold(value = node.alpha),
        values_samples=node.values_samples,
        input_length=node.input_length,
    )

def pad_to_depth(ensemble, target_depth):
    pestimator = deepcopy(ensemble)
    for idx_t, tree in enumerate(pestimator.flat_trees):
        idx = tree.n_nodes
        for idx_l, leaf in enumerate(tree.leaves):
            if (target_depth - leaf.depth) > 0:
                subtree_size = 2 ** (target_depth - leaf.depth + 1) - 1
                n_leaves_subtree = 2 ** (target_depth - leaf.depth)
                n_nodes_subtree = subtree_size - n_leaves_subtree
                # Create the dummy nodes
                dummies = list()
                for j in range(n_nodes_subtree):
                    nodo = create_dummy_node(name=j + idx, node=leaf.parent, dummy=True)
                    dummies.append(nodo)
                idx += n_nodes_subtree
                # Create the new leaves
                new_leaves = [
                    create_dummy_node(idx + _, leaf, dummy=_ > 0)
                    for _ in range(n_leaves_subtree)
                ]
                idx += n_leaves_subtree
                nodes_made = dummies + new_leaves

                # Unlink leaves from parents
                parent = leaf.parent
                shift_nodes(tree, [leaf.path_name], [None])

                # Link the new elements
                for level in range(0, subtree_size):
                    if (2 * level + 1) < len(nodes_made):
                        nodes_made[level].left = nodes_made[2 * level + 1]
                    if (2 * level + 2) < len(nodes_made):
                        nodes_made[level].right = nodes_made[2 * level + 2]
                if parent.left is None:
                    parent.left = nodes_made[0]
                else:
                    parent.right = nodes_made[0]


    return pestimator