import pytest
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from eden.frontend.sklearn import parse_random_forest
from eden.model.ensemble import Ensemble


def test_parse_random_forest_regression():
    boston = fetch_california_housing()
    model = RandomForestRegressor(n_estimators=3, random_state=0, max_depth=4)
    model.fit(boston.data, boston.target)

    result = parse_random_forest(model=model)
    assert isinstance(result, Ensemble)
    assert result.task == "regression"
    assert len(result.trees) == len(model.estimators_)
    assert result.input_length == model.n_features_in_
    assert result.output_length == 1


def test_parse_random_forest_binary_classification():
    iris = load_iris()
    binary_iris_target = [1 if i == 0 else 0 for i in iris.target]
    model = RandomForestClassifier(n_estimators=3, random_state=0, max_depth=4)
    model.fit(iris.data, binary_iris_target)

    result = parse_random_forest(model=model)
    assert isinstance(result, Ensemble)
    assert result.task == "classification_multiclass"
    assert len(result.trees) == len(model.estimators_)
    assert result.input_length == model.n_features_in_
    assert result.output_length == 2


def test_parse_random_forest_multiclass_classification():
    iris = load_iris()
    model = RandomForestClassifier(n_estimators=3, random_state=0, max_depth=4)
    model.fit(iris.data, iris.target)

    result = parse_random_forest(model=model)
    assert isinstance(result, Ensemble)
    assert result.task == "classification_multiclass"
    assert len(result.trees) == len(model.estimators_)
    assert result.input_length == model.n_features_in_
    assert result.output_length == len(set(iris.target))
