import numpy as np


class Tree:
    # Idx in the node array where stuff is stored
    FEATURE_IDX = 0
    THRESHOLD = 1
    RIGHT_CHILD = 2
    LEAF_FIELD = 2
    N_FIELDS = 3

    def __init__(self, decision_tree):
        self.tree = decision_tree.tree_
        self.nodes, self.leaves_idx, self.leaves = self._extract_tree(self.tree)

    def export(self):
        return self.nodes, self.leaves_idx, self.leaves

    def bfs_visit(self, tree):
        lc = tree.children_left
        rc = tree.children_right
        stack = list()
        bfs = list()
        stack.append(0)  # Root
        while len(stack) > 0:
            idx_nodo = stack.pop()
            if idx_nodo >= 0:
                stack.insert(0, lc[idx_nodo])
                stack.insert(0, rc[idx_nodo])
                bfs.append(idx_nodo)
        return np.asarray(bfs)

    def preorder_visit(self, tree):
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

    def _extract_tree(self, tree):
        pre_order = self.preorder_visit(tree)
        n_nodes = tree.node_count
        nodes_idx = tree.feature[pre_order] != -2
        leaves_idx = ~nodes_idx
        nodes = np.zeros(shape=(n_nodes, 3))
        nodes[:, self.FEATURE_IDX] = tree.feature[pre_order]
        nodes[:, self.THRESHOLD] = tree.threshold[pre_order]
        nodes[:, self.RIGHT_CHILD] = tree.children_right[pre_order]
        # Transforms the index of the right child in a shift.
        for i in range(nodes.shape[0]):
            if nodes[i, self.FEATURE_IDX] >= 0:
                nodes[i, self.RIGHT_CHILD] -= i
        # Extract the leaves
        foglie = tree.value[pre_order]
        leaves = np.concatenate(foglie[leaves_idx])
        if leaves.shape[-1] > 1:
            leaves = leaves / (leaves.sum(axis=1)[:, None])
        nodes[leaves_idx, self.RIGHT_CHILD] = np.arange(len(leaves))
        return nodes, leaves_idx, leaves
