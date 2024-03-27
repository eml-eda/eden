import pandas as pd
import numpy as np
from eden.model.node import Node
from bigtree import find
from sklearn.tree._tree import TREE_LEAF


def parse_tree(tree_df, column_names, input_length):
    root = None
    # It goes by-level, so the first node is always the root
    for idx, row in tree_df.iterrows():
        name = row.node_index
        parent_index = row.parent_index
        values = np.array([row.value]).reshape(1, -1)
        feature = column_names.index(row.split_feature) if row.split_feature is not None else input_length
        alpha = row.threshold if not np.isnan(row.threshold)  else TREE_LEAF
        values_samples = None
        if parent_index is None:
            root = Node(name = name, feature = feature, alpha = alpha, input_length = input_length, values = values, values_samples = values_samples)
        else:
            parent_node = find(root, lambda node : node.name == parent_index)
            parent_row = tree_df[tree_df.node_index == parent_index].iloc[0].to_dict()
            if parent_row["left_child"] == name:
                parent_node.left = Node(name = name, feature = feature, alpha = alpha, input_length = input_length, values = values, values_samples = values_samples)
            else:
                parent_node.right = Node(name = name, feature = feature, alpha = alpha, input_length = input_length, values = values, values_samples = values_samples)
    return root

