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
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import shutil


def plot_ensemble(model, path):
    path = os.path.join(path, "logs")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(name=path, exist_ok=True)
    # Also store the test inputs
    np.savetxt(os.path.join(path, "input-data.txt"), model.X_test_, fmt="%+i")
    if hasattr(model.estimator, "estimators_"):
        for idx, t in enumerate(model.estimator.estimators_):
            plot_tree(t)
            plt.savefig(os.path.join(path, f"tree_{idx}.png"))
            plt.clf()
    else:
        plot_tree(model.estimator)
        plt.savefig(os.path.join(path, f"tree_0.png"))
        plt.clf()


def export(
    *,
    eden_model: "EdenGarden",
    deployment_folder: str = "eden-ensemble",
    target: str = "all",
):
    eden_model = deepcopy(eden_model)
    os.makedirs(name=deployment_folder, exist_ok=True)
    # Plot each tree in ensemble using sklearn plotter
    plot_ensemble(eden_model, path=deployment_folder)
    # Sub-dir in data/
    for root, dirs, files in os.walk(DATA_DIR):
        if len(files) == 0:
            continue
        if (target != "all") and ("common" not in root and target not in root):
            # Skipped as it is for a different target
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
                try:
                    import clang_format

                    os.system(f"clang-format -i {os.path.join(tgt_dir,file_name)}")
                except:
                    pass
