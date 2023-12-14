import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from eden import EdenGarden, quantize, collapse_same_class_nodes
import subprocess

import pytest
from subprocess import STDOUT, check_output

@pytest.mark.parametrize("n_trees", [1, 2, 3, 4, 5], ids = [f"n_trees={i}" for i in range(1,6)])
@pytest.mark.parametrize("depth", [1, 2, 3, 4, 5], ids = [f"depth={i}" for i in range(1,6)])
def test_ensemble(n_trees, depth, request):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=0
    )
    X_min, X_max = X_train.min(), X_train.max()
    X_train = quantize(data=X_train, min_val=X_min, max_val=X_max, bits=16)
    X_test = quantize(data=X_test, min_val=X_min, max_val=X_max, bits=16)

    model = RandomForestClassifier(
        random_state=0, n_estimators=n_trees, max_depth=depth, n_jobs=10
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
    eden_model.deploy(
        deployment_folder="eden-ensemble", target="gcc", export_plot_trees=True
    )
    print("DEploy")

    for i in range(len(X_test)):
        print("Parsing input ", i)
        output = check_output(
            f"cd eden-ensemble/gcc && make all run INPUT_IDX={i} ", stderr=STDOUT, timeout=15, shell = True
        )


