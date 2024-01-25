from eden.frontend.sklearn import parse_random_forest
from eden.transform.quantization import (
    quantize_leaves,
    quantize_post_training_alphas,
    quantize_pre_training_alphas,
)
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np


def test_quantize_leaves():
    iris = load_iris()
    model = RandomForestClassifier(
        n_estimators=10, random_state=0, max_depth=7, min_samples_leaf=10
    )

    model.fit(iris.data, iris.target)

    mod = parse_random_forest(model=model)
    quantized = quantize_leaves(estimator=mod, precision=8)

    for t in quantized.trees:
        for leaf in t.leaves:
            assert leaf.values.dtype == np.uint8

    quantized = quantize_leaves(estimator=mod, precision=16)

    for t in quantized.trees:
        for leaf in t.leaves:
            assert leaf.values.dtype == np.uint16

    quantized = quantize_leaves(estimator=mod, precision=32)

    for t in quantized.trees:
        for leaf in t.leaves:
            assert leaf.values.dtype == np.uint32
