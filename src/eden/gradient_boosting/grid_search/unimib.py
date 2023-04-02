from copy import deepcopy
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from eden.gradient_boosting.grid_search.config import (
    PARAMS,
    GBT_MAX_ESTIMATORS,
    LEAVES_BITS,
)
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import download_unimib_dataset, prepare_unimib_dataset
import pandas as pd
from copy import deepcopy
import multiprocessing as mp
from eden.utils import score
import joblib as jbl
from eden.utils import find_and_load

OUTPUT_FILENAME = f"unimib-gridsearch.csv"
N_PROCESSES = 24


def benchmark_setup(max_depth, bits_input):
    print("Got ", max_depth, bits_input)
    output = list()

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
        bits_input=bits_input,
        balanced_train=True,
    )
    clf = find_and_load(
        bits_input=bits_input,
        max_depth=max_depth,
        dataset="unimib",
        classifier="gbdt",
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
            filename=f"logs/gbdt/unimib/unimib-maxdepth{max_depth}-bitsinput{bits_input}-estimators{GBT_MAX_ESTIMATORS}.jbl",
        )

    for n_estimators in range(1, GBT_MAX_ESTIMATORS + 1):
        classifier = GBDT(
            base_model=deepcopy(clf), bits_input=bits_input, n_estimators=n_estimators
        )
        clf_data = classifier.export_to_dict()
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
    args = PARAMS
    print(args)
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
    grid_search()


if __name__ == "__main__":
    main()
