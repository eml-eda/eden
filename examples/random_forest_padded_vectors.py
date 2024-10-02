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

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from eden.transform.quantization import (
    quantize,
    quantize_pre_training_alphas,
)
from eden.frontend.sklearn.ensemble import parse_random_forest
from eden.transform.pruning import prune_same_class_leaves
from eden.model.ensemble import Ensemble
from scipy.stats import mode
from eden.transform.padding import pad_to_depth
from bigtree import tree_to_dot
from eden.export.fvectors import extract_fvectors_ensemble

np.random.seed(0)


INPUT_BITS = 8
OUTPUT_BITS = 8  # This is ignored and computed depending on the #Classes
N_TREES = 1
DEPTH = 6


def main():
    # Eden related part
    X, y = load_iris(return_X_y=True)
    X_min, X_max = X.min(), X.max()
    X, _, _ = quantize(data=X, min_val=X.min(), max_val=X.max(), precision=INPUT_BITS)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )

    model = RandomForestClassifier(
        random_state=0, n_estimators=N_TREES, max_depth=DEPTH, min_samples_leaf=4
    )
    model.fit(X_train, y_train)

    emodel: Ensemble = parse_random_forest(model=model)
    emodel = quantize_pre_training_alphas(
        estimator=emodel, precision=INPUT_BITS, min_val=X_min, max_val=X_max
    )

    # Voting
    emodel: Ensemble = prune_same_class_leaves(estimator=emodel)

    # Padding - change here the pad depth
    emodel = pad_to_depth(ensemble=emodel, target_depth=6)

    # Plot the tree, for debugging
    graph = tree_to_dot(emodel.flat_trees[0])
    graph.write_png("tree.png")

    # Change the C value from here
    extract_fvectors_ensemble(emodel, c=2)

    # CARE: the vectorization is not simulated
    # Convert the votes in qpredictions in the final predictions
    qpredictions = emodel.predict_raw(X).squeeze(-1)
    qmodes = mode(qpredictions, 1)
    qclasses = qmodes.mode

    predictions = model.predict(X)
    print("FP-Accuracy", accuracy_score(y, predictions))
    print("Q-Accuracy", accuracy_score(y, qclasses))


main()
