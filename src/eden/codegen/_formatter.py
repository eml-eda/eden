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


def to_node_struct(root_data, feature_idx_data, threshold_data, right_child_data):
    node_str = list()
    for tree_idx in range(len(root_data)):
        node_str.append(f"// Tree {tree_idx}")
        idx_start = root_data[tree_idx]
        if tree_idx == (len(root_data) - 1):
            idx_end = len(feature_idx_data)
        else:
            idx_end = root_data[tree_idx + 1]
        for f_idx, th, r_c in zip(
            feature_idx_data[idx_start:idx_end],
            threshold_data[idx_start:idx_end],
            right_child_data[idx_start:idx_end],
        ):
            th = int(th) if th.is_integer() else th
            node_str.append("{" + f"{f_idx}, {th}, {r_c}" + "}")
    node_str = ",\n".join(node_str)
    return node_str


def to_c_array(array, separator_string: Optional[str] = "// Tree"):
    array_str = list()
    if isinstance(array, np.ndarray) and len(array.shape) == 1:
        array = array.reshape(1, -1)
    for tree_idx in range(len(array)):
        if separator_string is not None:
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
