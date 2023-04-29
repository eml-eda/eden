import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(
    scope="module",
    params=[2, 3, 4, 5],
    ids=["binary", "multiclass(3)", "multiclass(4)", "multiclass(5)"],
)
def n_classes(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[125, 256],
    ids=["feature_idx(int8_t)", "feature_idx(in16_t)"],
)
def n_features(request):
    return request.param


@pytest.fixture(
    scope="module", params=[True, False], ids=["signed input", "unsigned input"]
)
def signed_input(request):
    return request.param


@pytest.fixture(scope="module")
def classification_dataset(n_classes, n_features, signed_input):
    X, y = make_classification(
        n_samples=30,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=n_features // 2,
        random_state=0,
    )
    if not signed_input:
        X += X.min()
    return X, y


@pytest.fixture(scope="module")
def regression_dataset(n_features, signed_input):
    X, y = make_regression(n_samples=30, n_features=n_features, random_state=0)
    if not signed_input:
        X += X.min()
    return X, y


@pytest.fixture(scope="function")
def get_regression_dataset(regression_dataset):
    X, y = regression_dataset
    return X.copy(), y.copy()


@pytest.fixture(scope="function")
def get_classification_dataset(classification_dataset):
    X, y = classification_dataset
    return X.copy(), y.copy()
