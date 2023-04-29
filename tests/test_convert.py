import pytest
from .fixtures.data import *
from .fixtures.models import *
from eden import convert, run
import numpy as np

EPS = 0.01


def test_convert_any(get_rf_regression):
    # Test data
    clf = get_rf_regression
    ensemble = convert(model=clf)
    output = run(ensemble=ensemble)
    test_data = np.zeros((1, ensemble.n_features))
    golden = (
        np.asarray([t.predict(test_data) for t in clf.estimators_]).sum(axis=0).item()
    )
    assert (golden - output) < EPS


def test_convert_any2(get_dt_regression):
    # Test data
    clf = get_dt_regression
    ensemble = convert(model=clf)
    output = run(ensemble=ensemble)
    test_data = np.zeros((1, ensemble.n_features))
    golden = clf.predict(test_data)
    assert (golden - output) < EPS
