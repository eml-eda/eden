import numpy as np
from typing import Optional, List, Union
import numpy as np


def ctype_to_vtype(ctype: str):
    vtype = None
    if ctype != "float":
        if ctype == "uint16_t":
            vtype = "v2u"
        elif ctype == "uint8_t":
            vtype = "v4u"
    return vtype


def nptype_to_ctype(*, dtype : np.dtype) -> str:
    if "float" in dtype.name:
        return "float"
    elif dtype == np.uint16:
        return "uint16_t"
    elif dtype == np.uint8:
        return "uint8_t"
    elif dtype == np.uint32:
        return "uint32_t"
    elif dtype == np.int16:
        return "int16_t"
    elif dtype == np.int8:
        return "int8_t"
    elif dtype == np.int32:
        return "int32_t"
    raise NotImplementedError(f"Type {dtype} not supported")


def to_node_struct(root_data, feature_idx_data, alpha_data, right_child_data):
    node_str = list()
    for tree_idx in range(len(root_data)):
        node_str.append(f"// Tree {tree_idx}")
        idx_start = root_data[tree_idx]
        if tree_idx == (len(root_data) - 1):
            idx_end = len(feature_idx_data)
        else:
            idx_end = root_data[tree_idx + 1]
        for f_idx, al, r_c in zip(
            feature_idx_data[idx_start:idx_end],
            alpha_data[idx_start:idx_end],
            right_child_data[idx_start:idx_end],
        ):
            al = int(al) if al.is_integer() else al
            node_str.append("{" + f".feature={f_idx}, .alpha={al}, .child_right={r_c}" + "}")
    node_str = ",\n".join(node_str)
    return node_str


def to_c_array(array, separator_string: Optional[str] = "// Tree"):
    array_str = list()
    if isinstance(array, np.ndarray) and len(array.shape) == 1:
        array = array.reshape(1, -1)
    for tree_idx in range(len(array)):
        if separator_string is not None:
            if separator_string != "":
                array_str.append(f"{separator_string} {tree_idx}")
            for el in array[tree_idx]:
                el = int(el) if el.is_integer() else el
                array_str.append(f"{el}")
    array_str = ",\n".join(array_str)
    return array_str


def to_c_array2d(array):
    array_str = list()
    for row in range(len(array)):
        s = ",".join(map(str, list(array[row])))
        s = "{" + s + "}"
        array_str.append(s)
    array_str = ",\n".join(array_str)
    return array_str
