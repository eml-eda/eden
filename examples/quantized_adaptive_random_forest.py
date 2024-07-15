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
Simple example showing how to:
1. Benchmark the quantized model in python
2. Try an adaptive approach

"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from eden.transform.quantization import (
    quantize,
    quantize_leaves,
    quantize_post_training_alphas,
    quantize_pre_training_alphas,
)
from eden.inference import predict_adaptive, score_margin
from sklearn.metrics import accuracy_score
from eden.frontend.sklearn.ensemble import parse_random_forest


INPUT_BITS = 8
OUTPUT_BITS = 8

X, y = load_iris(return_X_y=True)
X, _, _ = quantize(data=X, min_val=X.min(), max_val=X.max(), precision=INPUT_BITS)
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
model.fit(X, y)

emodel = parse_random_forest(model=model)
qemodel = quantize_leaves(estimator=emodel, precision=INPUT_BITS)


qemodel = quantize_pre_training_alphas(
    estimator=qemodel,
    precision=INPUT_BITS,
    min_val=X.min(),
    max_val=X.max(),
)

qpredictions = qemodel.predict(X).transpose(1, 0, 2)
qclasses = qpredictions.sum(axis=0).argmax(axis=-1)

predictions = model.predict(X)
print("FP-Accuracy", accuracy_score(y, predictions))
print("Q-Accuracy", accuracy_score(y, qclasses))

###
# Adaptive
# We use again the quantized logits, but since we will use the aggregated metrics
# we need first to accumulate them
THRESHOLDS = np.asarray([0, 10, 20])
acc_qlogits = np.cumsum(qpredictions, axis=0)
# Also aggregated_score_margin can be called directly on the NOT aggregated logits
early_scores = score_margin(acc_qlogits)
adaptive_predictions, classifiers_used = predict_adaptive(
    predictions=acc_qlogits,
    thresholds=THRESHOLDS,  # Some random thresholds
    early_scores=early_scores,
)
for idx, th in enumerate(THRESHOLDS):
    print("Threshold : ", th)
    print(
        "Accuracy: ",
        accuracy_score(y, np.argmax(adaptive_predictions[idx], axis=-1)),
        "Mean Trees Used: ",
        classifiers_used[idx].mean(),
    )
    print("--" * 20)
