from .tree import parse_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from eden.model.node import Node
from typing import List
from eden.model.aggregation import sum
from eden.model.ensemble import Ensemble


def parse_random_forest(*, model, aggregate_function=sum):
    assert isinstance(
        model, (RandomForestClassifier, RandomForestRegressor)
    ), "Model must be a RandomForestClassifier or RandomForestRegressor"
    assert hasattr(model, "estimators_"), "Model must be fitted"

    task = (
        "classification_multiclass"
        if isinstance(model, RandomForestClassifier)
        else "regression"
    )
    output_length: int = (
        model.n_classes_ if isinstance(model, RandomForestClassifier) else 1
    )
    input_length: int = model.n_features_in_

    trees: List[Node] = list()
    for tree in model.estimators_:
        trees.append(parse_tree(tree))
    ensemble = Ensemble(
        trees=trees,
        task=task,
        input_length=input_length,
        output_length=output_length,
        aggregate_function=aggregate_function,
    )
    return ensemble
