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
from eden.optimization import quantize, quantize_alphas, get_output_range
from eden.inference import predict_adaptive, score_margin
from sklearn.metrics import accuracy_score


INPUT_BITS = 8
OUTPUT_BITS = 8

X, y = load_iris(return_X_y=True)
model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
# Quantization aware training
X = quantize(data=X, min_val=X.min(), max_val=X.max(), bits=INPUT_BITS)
model.fit(X, y)

# (Optional) Quantize the thresholds (for QAT we just apply a ceil function)
model = quantize_alphas(
    estimator=model,
    input_min_val=X.min(),
    input_max_val=X.max(),
    method="qat",
    bits=INPUT_BITS,
)

# Output quantization (post-training) is less automatized, but still fast.
# N.B It should be applied only on the ACCUMULATED probabilities
# (not the mean probability) for RFs and on the output of the decision function for GBT.
output_min, output_max = get_output_range(estimator=model)
# We can obtain the raw logits for RFs in the following way:
logits = np.asarray([tree.predict_proba(X) for tree in model.estimators_])
qlogits = quantize(
    data=logits,
    min_val=output_min,
    max_val=output_max,
    bits=OUTPUT_BITS,
    method="trunc",
)

qpredictions = np.argmax(np.sum(qlogits, axis=0), axis=-1)

# We get the original predictions for comparison.
predictions = model.predict(X)
print("FP-Accuracy", accuracy_score(y, predictions))
print("Q-Accuracy", accuracy_score(y, qpredictions))

###
# Adaptive
# We use again the quantized logits, but since we will use the aggregated metrics
# we need first to accumulate them
THRESHOLDS = np.asarray([0, 10, 20])
acc_qlogits = np.cumsum(qlogits, axis=0)
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
