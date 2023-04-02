import pandas as pd
import os
from sklearn.ensemble import GradientBoostingClassifier
from eden.gradient_boosting.gbdt import GBDT
from eden.datasets import prepare_ninapro_dataset
import joblib as jbl
import numpy as np
from eden.utils import pareto, infer_objects
from eden.inference.ensemble2template import Ensemble2Template
from eden.inference.profiler import Profiler
from copy import deepcopy
from eden.inference.utils import deployment_subset
import json
from eden.utils import find_and_load

N_JOBS = 12
FAST_MODE = True
ADAPTIVE_MODEL = "results/gbdt/adaptive/ninapro-adaptive-baseline.json"
fpath = "results/gbdt/gridsearch/ninapro-gridsearch.csv"


def deploy_grid(fpath_grid_search):
    stats_architetture = list()
    output_fname = os.path.basename(fpath_grid_search)
    output_fname, _ = os.path.splitext(output_fname)
    output_fname += "-deployed-pareto.csv"

    # Preparo i dati
    models = pd.read_csv(fpath_grid_search)

    # Escludo i float
    models = models[~models.bits_leaves.isna()]
    models = models[~models.bits_inputs.isna()]
    models = infer_objects(models)

    # Per ogni paziente....
    for patient in range(1, 27 + 1):
        # Estraggo i pareto - Validation Set - Balanced Accuracy
        models_patient = models.copy()
        models_patient = models_patient[models_patient.patient == patient]
        models_patient = pareto(
            models_patient,
            "n_branches_val",
            "balanced_accuracy_val",
            additional_columns=["Memory[kB]"],
            additional_orders=[True],
        )

        for model in [*models_patient.itertuples()][::-1]:
            model_memory = models_patient.loc[model.Index, "Memory[kB]"]
            if model_memory > 500:
                print(f"Model {model.Index} is too large")

            print(f"Training model {model.Index}")

            (
                X_train,
                X_val,
                X_test,
                y_train,
                y_val,
                y_test,
                _,
            ) = prepare_ninapro_dataset(
                bits_input=int(model.bits_inputs), patient=patient, balanced_train=True
            )
            clf = find_and_load(
                classifier="gbdt",
                bits_input=int(model.bits_inputs),
                max_depth=int(model.depth) - 1,
                dataset="ninapro",
                patient=patient,
                n_estimators=int(model.n_estimators),
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
            X_test = deployment_subset(classifier, X_test, n_samples_out=250)
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
            dati_modello = models_patient.loc[model.Index, :]
            stats["Architettura"] = model.Index
            stats = stats.assign(**dati_modello.to_dict()).mean()
            # print(stats)
            stats_architetture.append(stats)
            csv_architetture = pd.DataFrame(stats_architetture)
            csv_architetture.to_csv(output_fname, index=False)
            if "Cycles" in stats and FAST_MODE:
                break

    top_model = dict()
    for patient in range(1, 27 + 1):
        csv_architetture_p = csv_architetture[csv_architetture["patient"] == patient]
        deployed_arcs = csv_architetture_p[~csv_architetture_p["Cycles"].isna()]
        top_model_arch = deployed_arcs.iloc[
            deployed_arcs["balanced_accuracy_val"].argmax(), :
        ]["Architettura"]
        # Le prendo tutte
        top_model_series = deployed_arcs[
            deployed_arcs["Architettura"] == top_model_arch
        ].mean()
        top_model[patient] = top_model_series.to_dict()

    output_fname_json = "ninapro-adaptive-baseline.json"
    with open(output_fname_json, "w") as fp:
        json.dump(top_model, fp, indent=4)


def deploy_adaptive():
    model_json = json.load(open(ADAPTIVE_MODEL, "r"))

    for patient in range(1, 27 + 1):
        print(patient)
        model = pd.Series(model_json[str(patient)])
        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
            _,
        ) = prepare_ninapro_dataset(
            bits_input=model.bits_inputs, balanced_train=True, patient=patient
        )

        clf = find_and_load(
            classifier="gbdt",
            n_estimators=int(model.n_estimators),
            bits_input=int(model.bits_inputs),
            max_depth=int(model.depth) - 1,
            patient=patient,
            dataset="ninapro",
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
        X_test = deployment_subset(classifier, X_test, n_samples_out=250)
        template.set_input_data(X_test)
        template.write_ensemble()
        profiler = Profiler()
        print("Deploying...")

        model_json[str(patient)]["adaptive-profiling"] = {}
        for policy in ["aggregated-max", "aggregated-score-margin"]:
            if policy == "aggregated-max":
                mode = "-DDYNAMIC -DMAX_MARGIN"
            else:
                mode = "-DDYNAMIC -DSCORE_MARGIN"
            model_json[str(patient)]["adaptive-profiling"][policy] = {}
            for batch in [1, 2, 4, 8]:
                model_json[str(patient)]["adaptive-profiling"][policy][batch] = {}
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
                    model_json[str(patient)]["adaptive-profiling"][policy][batch][
                        min(stopping_group * batch, int(model.n_estimators))
                    ] = stats.mean().to_dict()
            with open("ninapro-adaptive-baseline-deployed.json", "w") as f:
                json.dump(model_json, f, indent=4)


def merge():
    print("Merge")
    BATCH = 4
    with open("ninapro-adaptive-baseline-deployed.json", "r") as f:
        to_ = json.load(f)

    output_fname_json = "ninapro-adaptive-baseline.json"
    with open(output_fname_json, "r") as fp:
        from_ = json.load(fp)

    for patient in to_.keys():
        to_[patient][f"Cycles{BATCH}"] = from_[patient]["Cycles"]

    with open("ninapro-adaptive-baseline-deployed.json", "w") as f:
        json.dump(to_, f, indent=4)


if __name__ == "__main__":
    deploy_grid(fpath)
    merge()
    # deploy_adaptive()
