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

from typing import Mapping, List
import numpy as np
import os
from ._formatter import to_c_array2d, to_c_array, to_node_struct


def _write_template_input(
    lookup, output_dir, template_data_src: Mapping, test_data: List[np.ndarray]
):
    os.makedirs(os.path.join(output_dir, "autogen"), exist_ok=True)
    output_fname = os.path.join(output_dir, "autogen", "eden_input.h")
    template = lookup.get_template("eden_input.h")

    t = template.render(input_list=[to_c_array(t) for t in test_data])

    with open(f"{output_fname}", "w") as out_file:
        out_file.write(t)
    file_extension = os.path.splitext(output_fname)[1]
    if file_extension in [".c", ".h"]:
        os.system(f"clang-format -i {output_fname}")
    return template_data_src
