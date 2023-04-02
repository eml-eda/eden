import pandas as pd
import json
from eden.datasets import prepare_hdd
import numpy as np
from eden.qwyc.qwyc import QWYC
from eden.utils import find_and_load
from copy import deepcopy
from eden.gradient_boosting.gbdt import GBDT
from eden.random_forest.rf import RF
from eden.utils import score


def rf():
    TOP_MODEL = json.load(open("results/rf/adaptive/hdd-adaptive-baseline-rf.json"))
    # Get the best model
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_hdd(
        bits_input=TOP_MODEL["bits_inputs"],
    )
    clf = find_and_load(
        classifier="rf",
        n_estimators=int(TOP_MODEL["n_estimators"]),
        bits_input=int(TOP_MODEL["bits_inputs"]),
        max_depth=None,
        dataset="hdd",
    )
    assert clf is not None
    classifier = RF(
        base_model=deepcopy(clf),
        bits_input=int(TOP_MODEL["bits_inputs"]),
        n_estimators=int(TOP_MODEL["n_estimators"]),
    )

    data_out = list()
    for a in np.logspace(-4, 0, 30):
        model_stats = deepcopy(TOP_MODEL)
        qw = QWYC(
            base_model=classifier,
            alpha=a,
            tolerance=1,
            leaves_bits=int(TOP_MODEL["bits_leaves"]),
        )
        qw.fit_no_ordering(X_val)
        preds, stop, n_branch = qw.predict(X_test)
        scores = score(y_test, preds)
        scores = {"adaptive_" + k: v for k, v in scores.items()}
        result = {
            "thr": a,
            "adaptive_n_estimators": stop.mean(),
            "adaptive_n_trees": stop.mean(),
            "adaptive_branches": n_branch.mean(),
            "method": "qwyc",
        }
        result.update(model_stats)
        result.update(scores)
        data_out.append(result)

        # Ordered
        qw = QWYC(
            base_model=classifier,
            alpha=a,
            tolerance=1,
            leaves_bits=int(TOP_MODEL["bits_leaves"]),
        )
        qw.fit(X_val)
        preds, stop, n_branch = qw.predict(X_test)
        scores = score(y_test, preds)
        scores = {"adaptive_" + k: v for k, v in scores.items()}
        result = {
            "thr": a,
            "adaptive_n_estimators": stop.mean(),
            "adaptive_n_trees": stop.mean(),
            "adaptive_branches": n_branch.mean(),
            "method": "qwyc-ordered",
        }
        result.update(model_stats)
        result.update(scores)
        data_out.append(result)
    df = pd.DataFrame(data_out)
    df.to_csv("hdd-adaptive-qwyc-rf.csv", index=False)


def gbdt():
    TOP_MODEL = json.load(open("results/gbdt/adaptive/hdd-adaptive-baseline.json"))
    FILEPATH = "results/gbdt/gridsearch/hdd-gridsearch.csv"
    # Get the best model
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_hdd(
        bits_input=TOP_MODEL["bits_inputs"],
    )
    clf = find_and_load(
        classifier="gbdt",
        n_estimators=int(TOP_MODEL["n_estimators"]),
        bits_input=int(TOP_MODEL["bits_inputs"]),
        max_depth=int(TOP_MODEL["depth"]) - 1,
        dataset="hdd",
    )
    assert clf is not None
    classifier = GBDT(
        base_model=deepcopy(clf),
        bits_input=int(TOP_MODEL["bits_inputs"]),
        n_estimators=int(TOP_MODEL["n_estimators"]),
    )

    data_out = list()
    for a in np.logspace(-4, 0, 30):
        model_stats = deepcopy(TOP_MODEL)
        qw = QWYC(
            base_model=classifier,
            alpha=a,
            tolerance=1,
            leaves_bits=int(TOP_MODEL["bits_leaves"]),
        )
        qw.fit_no_ordering(X_val)
        preds, stop, n_branch = qw.predict(X_test)
        scores = score(y_test, preds)
        scores = {"adaptive_" + k: v for k, v in scores.items()}
        result = {
            "thr": a,
            "adaptive_n_estimators": stop.mean(),
            "adaptive_n_trees": stop.mean(),
            "adaptive_branches": n_branch.mean(),
            "method": "qwyc",
        }
        result.update(model_stats)
        result.update(scores)
        data_out.append(result)

        # Ordered
        qw = QWYC(
            base_model=classifier,
            alpha=a,
            tolerance=1,
            leaves_bits=int(TOP_MODEL["bits_leaves"]),
        )
        qw.fit(X_val)
        preds, stop, n_branch = qw.predict(X_test)
        scores = score(y_test, preds)
        scores = {"adaptive_" + k: v for k, v in scores.items()}
        result = {
            "thr": a,
            "adaptive_n_estimators": stop.mean(),
            "adaptive_n_trees": stop.mean(),
            "adaptive_branches": n_branch.mean(),
            "method": "qwyc-ordered",
        }
        result.update(model_stats)
        result.update(scores)
        data_out.append(result)
    df = pd.DataFrame(data_out)
    df.to_csv("hdd-adaptive-qwyc.csv", index=False)


if __name__ == "__main__":
    rf()
    gbdt()
