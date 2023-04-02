from copy import deepcopy
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from eden.gradient_boosting.grid_search.config import (
    PARAMS,
    GBT_MAX_ESTIMATORS,
    LEAVES_BITS,
)
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import prepare_hdd
import pandas as pd
from copy import deepcopy
import multiprocessing as mp
from eden.utils import score
import joblib as jbl
from eden.utils import find_and_load

N_PROCESSES = 4
OUTPUT_FILENAME = f"hdd-gridsearch.csv"
GBT_MAX_ESTIMATORS = 50

"""
Funzione di appoggio per parallelizzare
"""


def benchmark_setup(max_depth, bits_input):
    print("Got ", max_depth, bits_input)
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

    clf = find_and_load(
        classifier="gbdt",
        bits_input=bits_input,
        max_depth=max_depth,
        dataset="hdd",
        n_estimators=GBT_MAX_ESTIMATORS,
    )
    if clf is None:
        clf = GradientBoostingClassifier(
            random_state=0,
            n_estimators=GBT_MAX_ESTIMATORS,
            max_depth=max_depth,
            init="zero",
        )
        clf.fit(X_train, y_train)
        jbl.dump(
            value=clf,
            filename=f"logs/gbdt/hdd/hdd-maxdepth{max_depth}-bitsinput{bits_input}-estimators{GBT_MAX_ESTIMATORS}.jbl",
        )
    # Complexities calcolate su ensemble piu' grande
    classifier = GBDT(
        base_model=deepcopy(clf),
        bits_input=bits_input,
        n_estimators=GBT_MAX_ESTIMATORS,
    )
    n_branches_train = classifier.predict_complexity(X_train).mean(-1)
    n_branches_val = classifier.predict_complexity(X_val).mean(-1)
    n_branches_test = classifier.predict_complexity(X_test).mean(-1)

    for bits_leaves in LEAVES_BITS:
        classifier = GBDT(
            base_model=deepcopy(clf),
            bits_input=bits_input,
            n_estimators=GBT_MAX_ESTIMATORS,
        )
        y_hat_train = classifier.staged_predict(X_train, bitwidth=bits_leaves)
        y_hat_val = classifier.staged_predict(X_val, bitwidth=bits_leaves)
        y_hat_test = classifier.staged_predict(X_test, bitwidth=bits_leaves)

        for n_estimators in range(0, GBT_MAX_ESTIMATORS):
            classifier = GBDT(
                base_model=deepcopy(clf),
                bits_input=bits_input,
                n_estimators=n_estimators + 1,
            )
            row = classifier.export_to_dict()
            row["bits_leaves"] = bits_leaves
            row.update(
                {
                    "n_branches_train": n_branches_train[: (n_estimators + 1)].sum(),
                    "n_branches_val": n_branches_val[: (n_estimators + 1)].sum(),
                    "n_branches_test": n_branches_test[: (n_estimators + 1)].sum(),
                }
            )
            scores = score(y_train, y_hat_train[n_estimators])
            row.update({f"{k}_train": v for k, v in scores.items()})
            scores = score(y_val, y_hat_val[n_estimators])
            row.update({f"{k}_val": v for k, v in scores.items()})
            scores = score(y_test, y_hat_test[n_estimators])
            row.update({f"{k}_test": v for k, v in scores.items()})

            if bits_leaves is not None:
                row.update(classifier.get_deployment_memory(bits_leaves=bits_leaves))
            output.append(row)
    return output


def try_single():
    result = benchmark_setup(max_depth=30, bits_input=32)
    result = pd.DataFrame(result)
    result.to_csv("hdd-single.csv")


def grid_search():
    args = PARAMS

    # Debug sequentially
    # for a in args:
    #    benchmark_setup(*a)
    with mp.Pool(processes=N_PROCESSES) as p:
        results = p.starmap(benchmark_setup, list(args))
    result = list()
    for dr in results:
        result += dr
    result = pd.DataFrame(result)
    result.to_csv(OUTPUT_FILENAME, index=False)


def main():
    # try_single()
    grid_search()


if __name__ == "__main__":
    main()
