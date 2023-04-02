import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import prepare_hdd
import joblib as jbl
import numpy as np
from eden.utils import pareto, infer_objects
from eden.inference.ensemble2template import Ensemble2Template
from eden.inference.profiler import Profiler
from copy import deepcopy
from eden.inference.utils import deployment_subset
from eden.utils import find_and_load
from glob import glob as glob
import json

N_JOBS = 8
FAST_MODE = True

fpath = "results/gbdt/gridsearch/hdd-gridsearch.csv"
ADAPTIVE_MODEL = "results/gbdt/adaptive/hdd-adaptive-baseline.json"


def deploy_grid(fpath_grid_search):
    stats_architetture = list()
    temporal = "notemporal" not in fpath_grid_search
    raw = "raw" in fpath_grid_search
    output_fname = os.path.basename(fpath_grid_search)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-deployed-pareto.csv"

    # Preparo i dati
    models = pd.read_csv(fpath_grid_search)

    # Escludo i float
    models = models[~models["Memory[kB]"].isna()]
    models = models[~models.bits_inputs.isna()]
    models = infer_objects(models)

    # Estraggo i pareto - Validation Set - Balanced Accuracy
    models = pareto(
        models,
        "n_branches_val",
        "f1_val",
        additional_columns=["Memory[kB]"],
        additional_orders=[True],
    )

    for model in [*models.itertuples()][::-1]:
        model_memory = models.loc[model.Index, "Memory[kB]"]
        if model_memory > 500:
            print(f"Model {model.Index} is too large")
        print(f"Training model {model.Index}")
        print(model)
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _,
        ) = prepare_hdd(
            bits_input=int(model.bits_inputs),
        )

        clf = find_and_load(
            classifier="gbdt",
            n_estimators=int(model.n_estimators),
            bits_input=int(model.bits_inputs),
            max_depth=int(model.depth) - 1,
            dataset="hdd",
        )
        assert clf is not None
        # clf = GradientBoostingClassifier(
        #    random_state=0,
        #    n_estimators=int(model.n_estimators),
        #    max_depth=int(model.depth) - 1,
        #    init="zero",
        # )
        # clf.fit(X_train, y_train)
        classifier = GBDT(
            base_model=deepcopy(clf),
            bits_input=int(model.bits_inputs),
            n_estimators=int(model.n_estimators),
        )
        clf_data = classifier.export_to_dict()
        clf_data["bits_leaves"] = int(model.bits_leaves)
        radici, foglie, nodi = classifier.get_deployment_structures(
            bits_leaves=model.bits_leaves
        )
        template = Ensemble2Template(clf_data)
        template.set_ensemble_data(roots=radici, leaves=foglie, nodes=nodi)
        X_test = deployment_subset(classifier, X_test, n_samples_out=100)
        template.set_input_data(X_test)
        template.write_ensemble()
        profiler = Profiler()
        print("Deploying...")
        stats = profiler.run_parallel_ensemble(
            n_inputs=X_test.shape[0],
            compile_path=template.DEPLOYMENT_PATH,
            lib="nobuffer",
            c_args="",
            n_jobs=N_JOBS,
        )
        dati_modello = models.loc[model.Index, :]
        stats["Architettura"] = model.Index
        stats = stats.assign(**dati_modello.to_dict()).mean()
        stats_architetture.append(stats)
        csv_architetture = pd.DataFrame(stats_architetture)
        csv_architetture.to_csv(output_fname, index=False)
        if "Cycles" in stats and FAST_MODE:
            break
    deployed_arcs = csv_architetture[~csv_architetture["Cycles"].isna()]
    top_model_arch = deployed_arcs.iloc[deployed_arcs["f1_val"].argmax(), :][
        "Architettura"
    ]
    # Le prendo tutte
    top_model = deployed_arcs[deployed_arcs["Architettura"] == top_model_arch].mean()
    top_model = top_model.to_dict()

    output_fname_json = "hdd-adaptive-baseline.json"
    with open(output_fname_json, "w") as fp:
        json.dump(top_model, fp, indent=4)


def deploy_adaptive():
    model_json = json.load(open(ADAPTIVE_MODEL, "r"))
    model = pd.Series(model_json)
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        _,
    ) = prepare_hdd(
        bits_input=int(model.bits_inputs),
    )

    clf = find_and_load(
        classifier="gbdt",
        n_estimators=int(model.n_estimators),
        bits_input=int(model.bits_inputs),
        max_depth=int(model.depth) - 1,
        dataset="hdd",
    )
    assert clf is not None

    classifier = GBDT(
        base_model=deepcopy(clf),
        bits_input=int(model.bits_inputs),
        n_estimators=int(model.n_estimators),
    )
    clf_data = classifier.export_to_dict()
    clf_data["bits_leaves"] = int(model.bits_leaves)
    radici, foglie, nodi = classifier.get_deployment_structures(
        bits_leaves=model.bits_leaves
    )
    template = Ensemble2Template(clf_data)
    template.set_ensemble_data(roots=radici, leaves=foglie, nodes=nodi)
    X_test = deployment_subset(classifier, X_test, n_samples_out=100)
    template.set_input_data(X_test)
    template.write_ensemble()
    profiler = Profiler()
    print("Deploying...")

    model_json["adaptive-profiling"] = {}
    for policy in ["aggregated-max", "qwyc"]:
        if policy == "aggregated-max":
            mode = "-DDYNAMIC -DMAX_MARGIN"
        else:
            mode = "-DDYNAMIC -DQWYC"
        model_json["adaptive-profiling"][policy] = {}
        for batch in [1, 2, 4, 8]:
            model_json["adaptive-profiling"][policy][batch] = {}
            print("Batch", batch)
            if (model.n_estimators // batch) < 1:
                print("Batch", batch, "skipped")
                continue
            template.write_dynamic_config(batch=batch)
            maximum_steps = int(np.ceil(model.n_estimators / batch))
            for stopping_group in range(1, maximum_steps + 1):
                print(f"...Stopping at {stopping_group}")
                mode_local = mode + f" -DPROFILING={stopping_group}"
                stats = profiler.run_parallel_ensemble(
                    n_inputs=X_test.shape[0],
                    compile_path=template.DEPLOYMENT_PATH,
                    lib="nobuffer",
                    c_args=mode_local,
                    n_jobs=N_JOBS,
                )
                model_json["adaptive-profiling"][policy][batch][
                    min(stopping_group * batch, int(model.n_estimators))
                ] = stats.mean().to_dict()
        with open("hdd-adaptive-baseline-deployed.json", "w") as f:
            json.dump(model_json, f, indent=4)


if __name__ == "__main__":
    deploy_grid(fpath)
    # deploy_adaptive()
