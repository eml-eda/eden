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

# Simple script to show how to export a quantized Random Forest in C
# N.B Quantization is performed inside the EdenGarden class, if you want to
# benchmark the accuracy of the model, see the other example.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from eden import EdenGarden, quantize, collapse_same_class_nodes
from sklearn.metrics import accuracy_score


INPUT_BITS = 16
OUTPUT_BITS = 8  # This is ignored and computed depending on the #Classes
N_TREES = 3
DEPTH = 4


X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)
X_min, X_max = X_train.min(), X_train.max()
X_train = quantize(data=X_train, min_val=X_min, max_val=X_max, bits=INPUT_BITS)
X_test = quantize(data=X_test, min_val=X_min, max_val=X_max, bits=INPUT_BITS)


model = RandomForestClassifier(
    random_state=0, n_estimators=N_TREES, max_depth=DEPTH, min_samples_leaf=4
)
model.fit(X_train, y_train)


# Quantize the model (N.B alphas are quantized post-training)
eden_model = EdenGarden(
    estimator=model,
    input_range=(X_train.min(), X_train.max()),
    input_qbits=16,
    output_qbits=None,
    quantization_aware_training=True,
    store_class_in_leaves=True,
)
# Utility method to convert the forest in a C-like format.
# Use the argument X_test to specify custom input data to write in C.
eden_model.fit(X_test)

# Let's benchmark the accuracy after pruning
print(
    "Eden nodes:",
    eden_model.n_nodes_,
    "First 10 inputs results",
    model.predict(X_test)[:10],
)

# Write the template, change the deployment folder to avoid overwriting multiple ensembles
eden_model.deploy(deployment_folder="eden-ensemble", target="gcc")
# Now we can manually go inside the generated folder and run the Makefile of the
# target architecture
# N.B This step requires a working Pulpissimo/Gap8 toolchain or GCC for the "gcc"
# folder.
