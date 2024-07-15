from bigtree import BinaryNode, print_tree, preorder_iter
import numpy as np


class Node(BinaryNode):
    def __init__(
        self, name, values, alpha, feature, values_samples, input_length: int, **kwargs
    ):
        super().__init__(name, **kwargs)
        self.values: np.ndarray = values
        self.feature = feature
        self.alpha = alpha
        self.input_length = input_length

        # Unused atm, but could be useful for pruning
        self.values_samples = values_samples

    def predict(self, x):
        if self.is_leaf:
            return self.values
        else:
            if x[self.feature] <= self.alpha:
                return self.left.predict(x)
            else:
                return self.right.predict(x)

    @property
    def n_leaves(self):
        return len([*self.n_leaves])

    def export_to_arrays(self):
        # Inefficient, yet easy to program, a better way would be to make it recursive
        assert self.is_root, "Export starts only from the root node"
        preorder_nodes = list(preorder_iter(self))
        alphas, features, children_left, children_right, values = (
            np.zeros(len(preorder_nodes)),
            np.zeros(len(preorder_nodes)),
            np.zeros(len(preorder_nodes)),
            np.zeros(len(preorder_nodes)),
            np.zeros((len(preorder_nodes), self.values.shape[-1])),
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
