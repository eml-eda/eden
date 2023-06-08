# *--------------------------------------------------------------------------*
# * Copyright (c) 2023 Politecnico di Torino, Italy                          *
# * SPDX-License-Identifier: Apache-2.0                                      *
# *                                                                          *
# * Licensed under the Apache License, Version 2.0 (the "License");          *
# * you may not use this file except in compliance with the License.         *
# * You may obtain a copy of the License at                                  *
# *                                                                          *
# * http://www.apache.org/licenses/LICENSE-2.0                               *
# *                                                                          *
# * Unless required by applicable law or agreed to in writing, software      *
# * distributed under the License is distributed on an "AS IS" BASIS,        *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
# * See the License for the specific language governing permissions and      *
# * limitations under the License.                                           *
# *                                                                          *
# * Author: Francesco Daghero francesco.daghero@polito.it                    *
# *--------------------------------------------------------------------------*

from typing import Optional, List, Union
import numpy as np


def _get_ctype(bits: int, signed: bool = False):
    # Bits should be already 8, 16 or 32
    if bits is not None:
        return f"uint{int(bits)}_t"
    else:
        return "float"


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


def to_c_array(
    array: List[Union[np.ndarray, int]], separator_string: Optional[str] = "// Tree"
):
    array_str = list()
    if isinstance(array, np.ndarray) and len(array.shape) == 1:
        array = array.reshape(1, -1)
    for tree_idx in range(len(array)):
        if separator_string is not None:
            array_str.append(f"{separator_string} {tree_idx}")

        if isinstance(array[tree_idx], int):
            array_str.append(f"{array[tree_idx]}")
        else:
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
