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
import pkg_resources

TEMPLATES_DIR = pkg_resources.resource_filename("eden", "data/templates")


# Not so useful, it is a template only to print the dequantized logits automatically
def _write_template_main(lookup, output_dir, template_data_src: Mapping) -> Mapping:
    os.makedirs(os.path.join(output_dir, "autogen"), exist_ok=True)
    for e in os.listdir(os.path.join(TEMPLATES_DIR)):
        full_path = os.path.join(TEMPLATES_DIR, e)
        if os.path.isfile(full_path) and "main" in e:
            output_fname = os.path.join(output_dir, "autogen", e)
            template = lookup.get_template(e)
            t = template.render(
                output_qparams=template_data_src.get("output_qparams", None)
            )

            with open(f"{output_fname}", "w") as out_file:
                out_file.write(t)
            file_extension = os.path.splitext(output_fname)[1]
            if file_extension in [".c", ".h"]:
                os.system(f"clang-format -i {output_fname}")
    return template_data_src
