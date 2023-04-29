import pytest
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


@pytest.fixture(
    scope="module",
    params=[1, 2, 8],
    ids=["n_estimators(1)", "n_estimators(2)", "n_estimators(8)"],
)
def n_estimators(request):
    return request.param


# DecisionTreeRegressor
@pytest.fixture(scope="module")
def dt_regression(regression_dataset):
    X, y = regression_dataset
    clf = DecisionTreeRegressor(max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="function")
def get_dt_classification(dt_classification):
    clf = dt_classification
    return deepcopy(clf)


# RandomForestClassifier
@pytest.fixture(scope="module")
def dt_classification(classification_dataset):
    X, y = classification_dataset
    clf = DecisionTreeClassifier(max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="function")
def get_dt_regression(dt_regression):
    clf = dt_regression
    return deepcopy(clf)


# RandomForestRegressor
@pytest.fixture(scope="module")
def rf_regression(regression_dataset, n_estimators):
    X, y = regression_dataset
    clf = RandomForestRegressor(n_estimators=n_estimators, max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="function")
def get_rf_classification(rf_classification):
    clf = rf_classification
    return deepcopy(clf)


# RandomForestClassifier
@pytest.fixture(scope="module")
def rf_classification(classification_dataset, n_estimators):
    X, y = classification_dataset
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=2, random_state=0)
    clf.fit(X, y)
    return clf


@pytest.fixture(scope="function")
def get_rf_regression(rf_regression):
    clf = rf_regression
    return deepcopy(clf)
