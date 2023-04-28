from typing import Optional
import numpy as np


def to_node_struct(feature_idx_data, threshold_data, right_child_data):
    node_str = list()
    for tree_idx in range(len(feature_idx_data)):
        node_str.append(f"// Tree {tree_idx}")
        for f_idx, th, r_c in zip(
            feature_idx_data[tree_idx],
            threshold_data[tree_idx],
            right_child_data[tree_idx],
        ):
            node_str.append("{" + f"{f_idx}, {th}, {r_c}" + "}")
    node_str = ",\n".join(node_str)
    return node_str


def to_c_array(array: np.ndarray, separator_string: Optional[str] = "// Tree"):
    array_str = list()
    if len(array.shape) < 2:
        array = array.reshape(1, -1)
    for tree_idx in range(len(array)):
        if separator_string is not None:
            array_str.append(f"{separator_string} {tree_idx}")
        for el in array[tree_idx]:
            array_str.append(f"{el}")
    array_str = ",\n".join(array_str)
    return array_str


def to_c_array2d(array):
    array_str = list()
    for tree_idx in range(len(array)):
        array_str.append(f"// Tree {tree_idx}")
        for row in range(len(array[tree_idx])):
            s = ",".join(map(str, list(array[tree_idx][row])))
            s = "{" + s + "}"
            array_str.append(s)
    array_str = ",\n".join(array_str)
    return array_str
