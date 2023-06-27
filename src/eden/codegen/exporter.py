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

"""
Collection of functions to export the data from a valid dictionary to 
the mako one.
"""
import pathlib
from typing import Mapping, List
import numpy as np
import os
from mako.lookup import TemplateLookup
import pkg_resources
from copy import deepcopy
from eden.codegen import _formatter as formatter

DATA_DIR = pkg_resources.resource_filename("eden", "data")


def export(
    *,
    eden_model: "EdenGarden",
    deployment_folder: str = "eden-ensemble",
):
    eden_model = deepcopy(eden_model)
    os.makedirs(name=deployment_folder, exist_ok=True)
    # Sub-dir in data/
    for root, dirs, files in os.walk(DATA_DIR):
        if len(files) == 0:
            continue
        tgt_dir = root.replace(DATA_DIR, deployment_folder)
        tgt_dir = tgt_dir.replace("common", "")
        tgt_dir = tgt_dir.replace("examples", "")
        tgt_dir = pathlib.Path(tgt_dir)
        lookup = TemplateLookup(
            directories=[root],
            strict_undefined=True,
        )
        os.makedirs(tgt_dir, exist_ok=True)
        for file_name in files:
            template = lookup.get_template(file_name)
            template_string = template.render(data=eden_model, formatter=formatter)
            with open(os.path.join(tgt_dir, file_name), "w") as f:
                f.write(template_string)

            file_extension = os.path.splitext(file_name)[1]
            if file_extension in [".c", ".h"]:
                os.system(f"clang-format -i {os.path.join(tgt_dir,file_name)}")
