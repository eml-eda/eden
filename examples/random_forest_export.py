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
from eden.optimization import quantize, quantize_alphas, get_output_range
from eden.inference import predict_adaptive, score_margin
from sklearn.metrics import accuracy_score
from eden import EdenGarden


INPUT_BITS = 8
OUTPUT_BITS = 8

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=10, max_depth=4, random_state=0)
model.fit(X, y)

eden_model = EdenGarden(
    estimator=model,
    input_range=(X.min(), X.max()),
    input_qbits=8,
    output_qbits=8,
    quantization_aware_training=False,
)
eden_model.fit()
eden_model.deploy()
