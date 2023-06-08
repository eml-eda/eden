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

from typing import Mapping
import numpy as np
from ._formatter import to_c_array2d, to_c_array, to_node_struct
import os
from ._formatter import _get_ctype


def _write_template_config(lookup, output_dir, template_data_src: Mapping):
    os.makedirs(os.path.join(output_dir, "autogen"), exist_ok=True)
    output_fname = os.path.join(output_dir, "autogen", "eden_cfg.h")
    template = lookup.get_template("eden_cfg.h")
    t = template.render(
        roots_ctype=_get_ctype(template_data_src["roots_bits"]),
        input_ctype=_get_ctype(template_data_src["input_bits"]),
        threshold_ctype=_get_ctype(template_data_src["input_bits"]),
        feature_ctype=_get_ctype(template_data_src["feature_bits"]),
        output_ctype=_get_ctype(template_data_src["output_bits"]),
        children_right_ctype=_get_ctype(template_data_src["children_right_bits"]),
        leaves_store_mode=template_data_src["leaves_store_mode"],
        n_estimators=template_data_src["n_estimators"],
        n_trees=template_data_src["n_trees"],
        leaf_len=template_data_src["leaf_len"],
        n_nodes=template_data_src["n_nodes"],
        input_len=template_data_src["input_len"],
        n_leaves=template_data_src["n_leaves"],
        output_len=template_data_src["output_len"],
        output_bits=template_data_src["output_bits"],
        cache=template_data_src["cache"],
    )

    with open(f"{output_fname}", "w") as out_file:
        out_file.write(t)
    file_extension = os.path.splitext(output_fname)[1]
    if file_extension in [".c", ".h"]:
        os.system(f"clang-format -i {output_fname}")
    return template_data_src
