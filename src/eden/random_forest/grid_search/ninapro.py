import os
import logging
from copy import deepcopy
import joblib as jbl
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import multiprocessing as mp
from eden.random_forest.grid_search.config import (
    PARAMS,
    RF_MAX_ESTIMATORS,
    LEAVES_BITS,
)
from eden.random_forest.rf import RF
from eden.datasets import prepare_ninapro_dataset
from joblib import load
import pandas as pd
from eden.utils import score
from eden.utils import find_and_load

START_PATIENT = 1
END_PATIENT = 27
OUTPUT_FILENAME = f"ninapro-gridsearch-rf_{START_PATIENT}to{END_PATIENT}.csv"
N_PROCESSES = 8


def benchmark_setup(patient, max_depth, bits_input):
    print("Got ", patient, max_depth, bits_input)
    output = list()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_ninapro_dataset(
        bits_input=bits_input, patient=patient, balanced_train=True
    )
    clf = find_and_load(
        bits_input=bits_input,
        max_depth=max_depth,
        dataset="ninapro",
        classifier="rf",
        n_estimators=RF_MAX_ESTIMATORS,
        patient=patient,
    )
    if clf is None:
        print("Error")
        clf = RandomForestClassifier(
            random_state=0,
            n_estimators=RF_MAX_ESTIMATORS,
            max_depth=max_depth,
            n_jobs=1,
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)
        jbl.dump(
            value=clf,
            filename=f"logs/rf/ninapro/ninapro-patient{patient}-maxdepth{max_depth}-bitsinput{bits_input}-estimators{RF_MAX_ESTIMATORS}.jbl",
        )
    for n_estimators in range(1, RF_MAX_ESTIMATORS + 1):
        classifier = RF(
            base_model=deepcopy(clf), bits_input=bits_input, n_estimators=n_estimators
        )
        clf_data = classifier.export_to_dict()
        clf_data["patient"] = patient
        clf_data["n_branches_train"] = (
            classifier.predict_complexity(X_train).mean(-1).sum()
        )
        clf_data["n_branches_val"] = classifier.predict_complexity(X_val).mean(-1).sum()
        clf_data["n_branches_test"] = (
            classifier.predict_complexity(X_test).mean(-1).sum()
        )

        for bits_leaves in LEAVES_BITS:
            row = dict()
            row.update(clf_data)
            row["bits_leaves"] = bits_leaves
            y_hat_train = classifier.predict(X_train, bitwidth=bits_leaves)
            y_hat_val = classifier.predict(X_val, bitwidth=bits_leaves)
            y_hat_test = classifier.predict(X_test, bitwidth=bits_leaves)

            scores = score(y_train, y_hat_train)
            row.update({f"{k}_train": v for k, v in scores.items()})
            scores = score(y_val, y_hat_val)
            row.update({f"{k}_val": v for k, v in scores.items()})
            scores = score(y_test, y_hat_test)
            row.update({f"{k}_test": v for k, v in scores.items()})
            if bits_leaves is not None:
                row.update(classifier.get_deployment_memory(bits_leaves=bits_leaves))
            output.append(row)
    return output


def grid_search():
    patient_args = list()
    args = PARAMS
    for arg in args:
        for p in range(START_PATIENT, END_PATIENT + 1):
            patient_args.append((p, *arg))
    print(patient_args)

    # Debug sequentially
    # for a in patient_args:
    #    benchmark_setup(*a)
    with mp.Pool(processes=N_PROCESSES) as p:
        results = p.starmap(benchmark_setup, list(patient_args))
    result = list()
    for dr in results:
        result += dr
    result = pd.DataFrame(result)
    result.to_csv(OUTPUT_FILENAME, index=False)


def main():
    grid_search()


if __name__ == "__main__":
    main()
