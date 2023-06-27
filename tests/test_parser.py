# TODO: Currently working for the previous version of the package.
raise NotImplementedError
from sklearn.datasets import load_iris, load_diabetes
import pytest
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from eden.parser.sklearn import parse_estimator
import json
import pathlib

SUPPORTED_ENSEMBLES = [
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
]


# Classification
@pytest.mark.parametrize("n_classes", [2, 3], ids=["binary", "multiclass"])
def test_parse_classification_tree(n_classes, request):
    X, y = load_iris(return_X_y=True)
    if n_classes == 2:
        X = X[y != 2]
        y = y[y != 2]

    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    model.fit(X, y)
    tree_dict = parse_estimator(estimator=model)

    path = pathlib.Path(
        f"tests/models/classification_tree_{request.node.callspec.id}.json"
    )
    # with open(path, "w") as f:
    #    json.dump(tree_dict, f, indent=4)
    with open(path, "r") as f:
        golden_dict = json.load(f)
    assert golden_dict == tree_dict


# Regression
def test_parse_regression_tree():
    X, y = load_diabetes(return_X_y=True)
    model = DecisionTreeRegressor(max_depth=3, random_state=0)
    model.fit(X, y)
    tree_dict = parse_estimator(estimator=model)

    path = pathlib.Path(f"tests/models/regression_tree.json")
    # with open(path, "w") as f:
    #    json.dump(tree_dict, f, indent=4)
    with open(path, "r") as f:
        golden_dict = json.load(f)
    assert golden_dict == tree_dict


# All ensembles, as long as they are based on classical DTs
@pytest.mark.parametrize("ensemble", SUPPORTED_ENSEMBLES)
def test_ensemble(ensemble):
    if "Classifier" in ensemble.__name__:
        X, y = load_iris(return_X_y=True)
    else:
        X, y = load_diabetes(return_X_y=True)

    model = ensemble(n_estimators=2, random_state=0)
    model.fit(X, y)
    ensemble_dict = parse_estimator(estimator=model)
    path = pathlib.Path(f"tests/models/{ensemble.__name__.lower()}.json")
    # with open(path, "w") as f:
    #    json.dump(ensemble_dict, f, indent=4)
    with open(path, "r") as f:
        golden_dict = json.load(f)
    assert golden_dict == ensemble_dict
