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
EDEN: Efficient Decision tree Ensemble
======================================

eden is a python module that enables a fast deployment of tree-based models at the edge.

It converts a sklearn model into a custom model, eventually performing a post-training 
quantization of the tree fields.
Finally, it generates C templates of ensemble, allowing a fast deployment at the edge.
More info at https://github.com/eml-eda/eden.
"""

import pkg_resources
from .edengarden import EdenGarden
from eden.optimization.quantization import quantize, quantize_alphas
from eden.optimization.leaf_placer import collapse_same_class_nodes

__version__ = pkg_resources.get_distribution("eden").version
