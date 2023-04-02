from eden.random_forest.rf import RF
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import prepare_ninapro_dataset
import os
import pandas as pd
from eden.utils import pareto, infer_objects
from eden.utils import find_and_load
from copy import deepcopy
import json

top_model_gbt = json.load(open("results/gbdt/adaptive/ninapro-adaptive-baseline.json"))
top_model_rf = json.load(open("results/rf/adaptive/ninapro-adaptive-baseline-rf.json"))


def benchmark_gbdt(fpath):
    output_fname = os.path.basename(fpath)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-adaptive.csv"

    # Preparo i dati
    models = pd.read_csv(fpath)

    # Prendo solo i primi tre pazienti
    results_global = list()
    for patient in range(1, 27 + 1):
        print("Patient", patient)
        top_model = top_model_gbt[str(patient)]

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _,
        ) = prepare_ninapro_dataset(
            bits_input=int(top_model["bits_inputs"]),
            patient=patient,
            balanced_train=True,
        )
        clf = find_and_load(
            classifier="gbdt",
            n_estimators=int(top_model["n_estimators"]),
            bits_input=int(top_model["bits_inputs"]),
            max_depth=int(top_model["depth"]) - 1,
            dataset="ninapro",
            patient=patient,
        )
        assert clf is not None
        classifier = GBDT(
            base_model=deepcopy(clf),
            bits_input=int(top_model["bits_inputs"]),
            n_estimators=int(top_model["n_estimators"]),
        )

        step = 1
        if int(top_model["bits_leaves"]) == 32:
            step = 2**19
        elif int(top_model["bits_leaves"]) == 16:
            step = 2**5

        thresholds = [
            *range(
                0,
                2 ** (int(top_model["bits_leaves"])) + 1,
                step,
            )
        ]
        results = classifier.adaptive_benchmark(
            X=X_test,
            y=y_test,
            bitwidth=int(top_model["bits_leaves"]),
            thresholds=thresholds,
        )
        results = results.assign(**top_model)
        results_global.append(results)
        results_csv = pd.concat(results_global)
        results_csv.to_csv(output_fname, index=False)
    return results


def benchmark_rf(fpath):
    output_fname = os.path.basename(fpath)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-adaptive.csv"

    # Preparo i dati
    models = pd.read_csv(fpath)

    # Prendo solo i primi tre pazienti
    results_global = list()
    for patient in range(1, 27 + 1):
        print("Patient", patient)
        top_model = top_model_rf[str(patient)]

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _,
        ) = prepare_ninapro_dataset(
            bits_input=int(top_model["bits_inputs"]),
            patient=patient,
            balanced_train=True,
        )
        clf = find_and_load(
            classifier="rf",
            n_estimators=int(top_model["n_estimators"]),
            bits_input=int(top_model["bits_inputs"]),
            max_depth=int(top_model["depth"]) - 1,
            dataset="ninapro",
            patient=patient,
        )
        assert clf is not None
        classifier = RF(
            base_model=deepcopy(clf),
            bits_input=int(top_model["bits_inputs"]),
            n_estimators=int(top_model["n_estimators"]),
        )

        step = 1
        if int(top_model["bits_leaves"]) == 32:
            step = 2**19
        elif int(top_model["bits_leaves"]) == 16:
            step = 2**5

        thresholds = [
            *range(
                0,
                2 ** (int(top_model["bits_leaves"])) + 1,
                step,
            )
        ]
        results = classifier.adaptive_benchmark(
            X=X_test,
            y=y_test,
            bitwidth=int(top_model["bits_leaves"]),
            thresholds=thresholds,
        )
        results = results.assign(**top_model)
        results_global.append(results)
        results_csv = pd.concat(results_global)
        results_csv.to_csv(output_fname, index=False)
    return results


if __name__ == "__main__":
    file = "results/gbdt/gridsearch/ninapro-gridsearch.csv"
    benchmark_gbdt(file)
    file = "results/rf/gridsearch/ninapro-gridsearch-rf.csv"
    benchmark_rf(file)
