from eden.random_forest.rf import RF
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import download_unimib_dataset, prepare_unimib_dataset
import os
import pandas as pd
from eden.utils import pareto, infer_objects
from eden.utils import find_and_load
from copy import deepcopy
import json

# Architetture benchmark, da derivare con gli script sotto deployment
# Evitano di usare modelli troppo grandi.
top_model_gbt = json.load(open("results/gbdt/adaptive/unimib-adaptive-baseline.json"))
top_model_rf = json.load(open("results/rf/adaptive/unimib-adaptive-baseline-rf.json"))


def benchmark_gbdt(fpath):
    output_fname = os.path.basename(fpath)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-adaptive.csv"

    # Preparo i dati
    models = pd.read_csv(fpath)

    # Escludo i float
    models = models[~models.bits_leaves.isna()]
    models = models[~models.bits_inputs.isna()]
    models = infer_objects(models)

    path = download_unimib_dataset("data")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_unimib_dataset(
        path=path,
        bits_input=top_model_gbt["bits_inputs"],
        balanced_train=True,
    )
    clf = find_and_load(
        classifier="gbdt",
        n_estimators=int(top_model_gbt["n_estimators"]),
        bits_input=int(top_model_gbt["bits_inputs"]),
        max_depth=int(top_model_gbt["depth"]) - 1,
        dataset="unimib",
    )
    assert clf is not None
    classifier = GBDT(
        base_model=deepcopy(clf),
        bits_input=int(top_model_gbt["bits_inputs"]),
        n_estimators=int(top_model_gbt["n_estimators"]),
    )

    step = 1
    if int(top_model_gbt["bits_leaves"]) == 32:
        step = 2**16
    elif int(top_model_gbt["bits_leaves"]) == 16:
        step = 2**3

    thresholds = [
        *range(
            0,
            2 ** (int(top_model_gbt["bits_leaves"])) + 1,
            step,
        )
    ]
    results = classifier.adaptive_benchmark(
        X=X_test,
        y=y_test,
        bitwidth=int(top_model_gbt["bits_leaves"]),
        thresholds=thresholds,
    )
    results = results.assign(**top_model_gbt)
    results.to_csv(output_fname, index=False)
    return results


def benchmark_rf(fpath):
    output_fname = os.path.basename(fpath)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-adaptive.csv"

    # Preparo i dati
    models = pd.read_csv(fpath)

    # Escludo i float
    models = models[~models.bits_leaves.isna()]
    models = models[~models.bits_inputs.isna()]
    models = infer_objects(models)

    path = download_unimib_dataset("data")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_unimib_dataset(
        path=path,
        bits_input=int(top_model_rf["bits_inputs"]),
        balanced_train=True,
    )
    clf = find_and_load(
        classifier="rf",
        n_estimators=int(top_model_rf["n_estimators"]),
        bits_input=int(top_model_rf["bits_inputs"]),
        max_depth=int(top_model_rf["depth"]) - 1,
        dataset="unimib",
    )
    assert clf is not None
    classifier = RF(
        base_model=deepcopy(clf),
        bits_input=int(top_model_rf["bits_inputs"]),
        n_estimators=int(top_model_rf["n_estimators"]),
    )
    step = 1
    if int(top_model_rf["bits_leaves"]) == 32:
        step = 2**16
    elif int(top_model_rf["bits_leaves"]) == 16:
        step = 2**3

    thresholds = [
        *range(
            0,
            2 ** (int(top_model_rf["bits_leaves"])) + 1,
            step,
        )
    ]

    results = classifier.adaptive_benchmark(
        X=X_test,
        y=y_test,
        bitwidth=int(top_model_rf["bits_leaves"]),
        thresholds=thresholds,
    )
    results = results.assign(**top_model_rf)
    results.to_csv(output_fname, index=False)
    return results


if __name__ == "__main__":
    # file = "results/gbdt/gridsearch/unimib-gridsearch.csv"
    # benchmark_gbdt(file)
    file = "results/rf/gridsearch/unimib-gridsearch-rf.csv"
    benchmark_rf(file)
