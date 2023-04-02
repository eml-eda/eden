from copy import deepcopy
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from eden.random_forest.grid_search.config import (
    PARAMS,
    RF_MAX_ESTIMATORS,
    LEAVES_BITS,
)
from eden.random_forest.rf import RF
from eden.datasets import prepare_hdd
import pandas as pd
from copy import deepcopy
import multiprocessing as mp
from eden.utils import score
import joblib as jbl
from eden.utils import find_and_load

N_PROCESSES = 1
RF_MAX_ESTIMATORS = 30
OUTPUT_FILENAME = f"hdd-gridsearch-rf.csv"


"""
Funzione di appoggio per parallelizzare
"""


def benchmark_setup(bits_input):
    print("Got ", bits_input)
    output = list()

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_hdd(bits_input=bits_input)
    clf = None
    # clf = find_and_load(
    #    bits_input=bits_input,
    #    max_depth=None,
    #    dataset="hdd",
    #    classifier="rf",
    #    n_estimators=RF_MAX_ESTIMATORS,
    # )
    # assert clf is not None
    if clf is None:
        clf = RandomForestClassifier(
            n_estimators=30,
            min_samples_split=10,
            # max_depth=15,
            random_state=3,
            n_jobs=10,
            # ccp_alpha=0.02,
        )
        clf.fit(X_train, y_train)
        jbl.dump(
            value=clf,
            filename=f"logs/rf/hdd/hdd-bitsinput{bits_input}-estimators{30}.jbl",
        )

    # Complexities calcolate su ensemble piu' grande
    classifier = RF(
        base_model=deepcopy(clf), bits_input=bits_input, n_estimators=RF_MAX_ESTIMATORS
    )
    n_branches_train = classifier.predict_complexity(X_train).mean(-1)
    n_branches_val = classifier.predict_complexity(X_val).mean(-1)
    n_branches_test = classifier.predict_complexity(X_test).mean(-1)

    for bits_leaves in [32]:
        classifier = RF(
            base_model=deepcopy(clf),
            bits_input=bits_input,
            n_estimators=RF_MAX_ESTIMATORS,
        )
        y_hat_train = classifier.staged_predict(X_train, bitwidth=bits_leaves)
        y_hat_val = classifier.staged_predict(X_val, bitwidth=bits_leaves)
        y_hat_test = classifier.staged_predict(X_test, bitwidth=bits_leaves)

        for n_estimators in range(1, RF_MAX_ESTIMATORS + 1):
            classifier = RF(
                base_model=deepcopy(clf),
                bits_input=bits_input,
                n_estimators=n_estimators,
            )
            row = classifier.export_to_dict()
            row["bits_leaves"] = bits_leaves
            row.update(
                {
                    "n_branches_train": n_branches_train[:(n_estimators)].sum(),
                    "n_branches_val": n_branches_val[:(n_estimators)].sum(),
                    "n_branches_test": n_branches_test[:(n_estimators)].sum(),
                }
            )
            scores = score(y_train, y_hat_train[n_estimators - 1])
            row.update({f"{k}_train": v for k, v in scores.items()})
            scores = score(y_val, y_hat_val[n_estimators - 1])
            row.update({f"{k}_val": v for k, v in scores.items()})
            scores = score(y_test, y_hat_test[n_estimators - 1])
            row.update({f"{k}_test": v for k, v in scores.items()})

            if bits_leaves is not None:
                row.update(classifier.get_deployment_memory(bits_leaves=bits_leaves))
            output.append(row)
    return output


def try_single():
    result = benchmark_setup(bits_input=16)
    result = pd.DataFrame(result)
    result.to_csv("hdd-single-rf.csv")


def grid_search():
    args = PARAMS
    args = [[32], [16], [8]]
    with mp.Pool(processes=N_PROCESSES) as p:
        results = p.starmap(benchmark_setup, list(args))
    result = list()
    for dr in results:
        result += dr
    result = pd.DataFrame(result)
    result.to_csv(OUTPUT_FILENAME, index=False)


def main():
    try_single()
    # grid_search()


if __name__ == "__main__":
    main()
