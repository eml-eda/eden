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
import shutil
from typing import Mapping, List
import numpy as np
import os
from mako.lookup import TemplateLookup
import pkg_resources
from copy import deepcopy
from ._data import _write_template_data
from ._config import _write_template_config
from ._main import _write_template_main
from ._input import _write_template_input

TEMPLATES_DIR = pkg_resources.resource_filename("eden", "data/templates")
LIB_DIR = pkg_resources.resource_filename("eden", "data/lib")
MAKEFILES_DIR = pkg_resources.resource_filename("eden", "data/makefiles")


def export(
    *, estimator_dict: Mapping, test_data: List[np.ndarray], output_dir: str = "."
):
    """
    Export the input estimator in C

    Parameters
    ----------
    estimator_dict : Mapping
        eden dictionary storing the information about the ensemble
    test_data : List[np.ndarray]
        test inputs to be dumped in C 
    output_dir : str, optional
        the directory where the C code is generated

    Returns
    -------
    Mapping
        The updated estimator dictionary
    """
    estimator_dict = deepcopy(estimator_dict)
    os.makedirs(name=output_dir, exist_ok=True)
    lookup = TemplateLookup(
        directories=[
            os.path.join(TEMPLATES_DIR),
        ],
        strict_undefined=True,
    )
    estimator_dict = _write_template_data(
        lookup=lookup, output_dir=output_dir, template_data_src=estimator_dict
    )
    estimator_dict = _write_template_config(
        lookup=lookup, output_dir=output_dir, template_data_src=estimator_dict
    )

    estimator_dict = _write_template_input(
        lookup=lookup,
        output_dir=output_dir,
        template_data_src=estimator_dict,
        test_data=test_data,
    )

    estimator_dict = _write_template_main(
        lookup=lookup, output_dir=output_dir, template_data_src=estimator_dict
    )
    _write_library(
        output_dir=output_dir,
    )
    return estimator_dict


def _write_library(output_dir: str):
    os.makedirs(os.path.join(output_dir, "eden"), exist_ok=True)
    for e in os.listdir(os.path.join(LIB_DIR)):
        full_path = os.path.join(LIB_DIR, e)
        os.makedirs(os.path.join(output_dir, "eden", "src"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "eden", "include"), exist_ok=True)
        if os.path.isfile(full_path):
            _, extension = os.path.splitext(e)
            if extension == ".h":
                subfolder = "include"
            elif extension == ".c":
                subfolder = "src"
            tgt_path = os.path.join(output_dir, "eden", subfolder, e)
            shutil.copy(full_path, tgt_path)
        else:
            tgt_path = os.path.join(output_dir, "eden", "include", e)
            shutil.copytree(full_path, tgt_path, dirs_exist_ok=True)

    for e in os.listdir(MAKEFILES_DIR):
        full_path = os.path.join(MAKEFILES_DIR, e)
        tgt_path = os.path.join(output_dir, e)
        if os.path.isfile(full_path):
            shutil.copy(full_path, tgt_path)
