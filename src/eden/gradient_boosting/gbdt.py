import numpy as np
from eden.tree import Tree
from eden.ensemble import Ensemble
from eden.utils import (
    quantize,
    adaptive_predict,
    compute_score_margin,
    compute_max_score,
    score,
    min_bits,
)
import pandas as pd
from eden.qwyc.qwyc import QWYC


class GBDT(Ensemble):
    def __init__(
        self, base_model, bits_input: int, n_estimators: int, order=None
    ) -> None:
        self.base_model = base_model
        # assert hasattr(self.base_model, "classes_"), "Base model is not fitted"
        self.n_estimators = n_estimators
        self.bits_input = bits_input
        if order is not None:
            self.reorder_estimators(order)

        # Extract stats from the base learner
        self.extract_stats_from_base()

        # Extract the structure
        (
            self.struct_nodi,
            self.struct_radici,
            self.struct_foglie,
            self.struct_foglie_albero,
        ) = self.extract_structure(ensemble=base_model, n_estimators=self.n_estimators)

        # Extract the stats from the structure
        self.extract_stats_from_structure(
            struct_nodi=self.struct_nodi,
            struct_radici=self.struct_radici,
            struct_foglie=self.struct_foglie,
            struct_foglie_albero=self.struct_foglie_albero,
        )

    def predict_complexity(self, X):
        """
        branch di ogni albero su tutti gli input
        """
        n_branches = np.zeros(shape=(self.n_estimators, self._n_classes, X.shape[0]))
        # For each estimator...
        for e_idx, estimator in enumerate(
            self.base_model.estimators_[: self.n_estimators]
        ):
            # For each tree in the estimators
            for c_idx, tree in enumerate(estimator):
                path = tree.decision_path(X).todense()
                n_branches[e_idx, c_idx, :] = path.sum(axis=-1).reshape(-1)
        return n_branches

    def predict_probs(self, X, bitwidth=None):
        # Somma gia' accumulata! Da trasformare.
        staged_raw_predictions_gen = self.base_model.staged_decision_function(X)
        staged_raw_predictions = np.stack([*staged_raw_predictions_gen], axis=0)[
            : self.n_estimators, :, :
        ]
        # golden = self.base_model.decision_function(X)
        if self.n_estimators > 1:
            staged_raw_predictions[1:] -= staged_raw_predictions[:-1]
        if bitwidth is not None:
            staged_raw_predictions = quantize(
                staged_raw_predictions, range=self.range_foglie, bitwidth=bitwidth
            )
        assert staged_raw_predictions.sum(axis=0).max() <= (2 ** (bitwidth - 1) - 1)
        assert staged_raw_predictions.sum(axis=0).min() >= -(2 ** (bitwidth - 1))
        return staged_raw_predictions

    def predict(self, X, bitwidth=None):
        predictions = np.sum(self.predict_probs(X, bitwidth=bitwidth), axis=0)
        predictions = (
            np.argmax(predictions, axis=-1) if self._n_classes != 1 else predictions > 0
        )
        return predictions

    def staged_predict(self, X, bitwidth=None):
        """
        returns [N_ESTIMATORS, N_SAMPLES]
        """
        predictions = np.cumsum(self.predict_probs(X, bitwidth=bitwidth), axis=0)
        predictions = (
            np.argmax(predictions, axis=-1) if self._n_classes != 1 else predictions > 0
        )
        return predictions

    def extract_stats_from_structure(
        self, struct_nodi, struct_radici, struct_foglie, struct_foglie_albero
    ):
        # Generali
        self.n_nodes = struct_nodi.shape[0]
        self.n_leaves = struct_foglie.shape[0]
        # Bits per struttura
        self.bits_roots = min_bits(self.n_nodes)
        self.bits_nodes_idx = min_bits(self.n_nodes)
        self.bits_leaves_idx = min_bits(self.n_leaves)
        self.bits_feature_idx = min_bits(self.n_features, has_negative_idx=True)
        self.bits_thresholds = self.bits_input
        self.bits_right_child = min_bits(struct_nodi[:, Tree.RIGHT_CHILD].max())

        # Quantizzazione
        classe = 0
        massimi_foglie = np.zeros(self._n_classes)
        minimi_foglie = np.zeros(self._n_classes)
        for idx_albero in range(self.n_trees):
            idx_prima_foglia = struct_foglie_albero[idx_albero]
            idx_ultima_foglia = (
                struct_foglie_albero[idx_albero + 1]
                if (idx_albero + 1) != self.n_trees
                else self.n_leaves
            )
            foglie_albero = struct_foglie[idx_prima_foglia:idx_ultima_foglia]
            massimi_foglie[classe] += foglie_albero.max()
            minimi_foglie[classe] += foglie_albero.min()
            classe = (classe + 1) % self._n_classes

        # TODO: Lavorare con i priors
        self.minimo_foglie = minimi_foglie.min()
        self.massimo_foglie = massimi_foglie.max()
        self.range_foglie = (self.minimo_foglie, self.massimo_foglie)

    def extract_structure(self, ensemble, n_estimators):
        """
        Iterate over the estimators and extract the trees data.
        """
        struct_nodi = np.zeros((0, Tree.N_FIELDS))
        struct_radici = np.zeros(self.n_trees, dtype=int)
        struct_foglie = np.zeros((0, 1))  # Changing from RF
        struct_foglie_albero = np.zeros((self.n_trees), dtype=int)
        trees_parsed = 0
        # For each estimator...
        for e_idx, estimator in enumerate(ensemble.estimators_[:n_estimators]):
            # For each tree in the estimators
            for c_idx, tree in enumerate(estimator):
                struct_radici[trees_parsed] = struct_nodi.shape[0]
                struct_foglie_albero[trees_parsed] = struct_foglie.shape[0]
                # Working with a DecisionTree from sklearn
                # Get the leaves values
                t = Tree(tree)
                nodes, leaves_idx, leaves = t.export()
                # Move the idx of the leaves according to the previous
                nodes[leaves_idx, Tree.LEAF_FIELD] += struct_foglie.shape[0]
                struct_nodi = np.concatenate([struct_nodi, nodes], axis=0)
                struct_foglie = np.concatenate([struct_foglie, leaves], axis=0)
                trees_parsed += 1
        return struct_nodi, struct_radici, struct_foglie, struct_foglie_albero

    def extract_stats_from_base(self):
        """
        Extract all useful information from the underline base model
        """
        # Sklearn data
        self.n_features = self.base_model.n_features_in_
        self.learning_rate = self.base_model.learning_rate
        self.n_classes = self.base_model.n_classes_
        self.depth = self.base_model.max_depth + 1
        # Additional variables
        self._n_classes = self.n_classes if self.n_classes > 2 else 1
        self.n_trees = self.n_estimators * self._n_classes

    def get_deployment_structures(self, bits_leaves):
        """
        Combina le strutture salvando le foglie nei nodi
        Quantizzazione: max(B_leaves, B_thresholds)
        """
        true_bits_leaves = max(bits_leaves, self.bits_thresholds)

        radici, foglie, nodi = self.get_structure(true_bits_leaves)
        leaves_idx = nodi[:, Tree.FEATURE_IDX] == -2
        nodi[leaves_idx, Tree.THRESHOLD] = foglie.reshape(-1)
        return radici, None, nodi

    def get_deployment_memory(self, bits_leaves):
        if bits_leaves is None or self.bits_thresholds is None:
            return {}
        threshold_field_bitwidth = max(bits_leaves, self.bits_thresholds)

        diz = {}
        diz["bits_node_struct"] = (
            threshold_field_bitwidth + self.bits_feature_idx + self.bits_right_child
        ) * self.n_nodes
        diz["bits_input_struct"] = self.bits_feature_idx * self.n_features
        diz["bits_output_struct"] = self._n_classes * threshold_field_bitwidth
        diz["bits_roots_struct"] = self.n_trees * self.bits_nodes_idx
        diz["bits_leaf_struct"] = 0
        diz["Memory[kB]"] = (
            diz["bits_node_struct"]
            + diz["bits_input_struct"]
            + diz["bits_output_struct"]
            + diz["bits_roots_struct"]
            + diz["bits_leaf_struct"]
        ) / (8 * 1000)
        return diz

    def compute_adaptive_method(
        self,
        method_name,
        logits,
        y,
        complexities,
        early_stop_scores,
        batch,
        cycles_data,
    ):
        def thresholds(x):
            t = np.unique(x)
            ts = np.unique(np.concatenate([t, t - 1]))
            print(f"Found {len(ts)} scores")
            while len(ts) >= 1000:
                ts = ts[::10]
            print(f"Used {len(ts)} scores")
            return ts

        results = list()
        for thr in thresholds(early_stop_scores):
            (
                adaptive_logits,
                adaptive_branches,
                stopping_trees,
            ) = adaptive_predict(
                logits=logits,
                branches=complexities,
                threshold=thr,
                early_scores=early_stop_scores,
            )
            stopping_trees = np.minimum(
                np.ones_like(stopping_trees) * self.n_estimators, stopping_trees * batch
            )
            predictions = (
                np.argmax(adaptive_logits, axis=-1)
                if self._n_classes != 1
                else adaptive_logits > 0
            )
            scores = score(y, predictions)
            scores = {"adaptive_" + k: v for k, v in scores.items()}
            result = {
                "thr": thr,
                "adaptive_n_estimators": stopping_trees.mean(),
                "adaptive_n_trees": stopping_trees.mean() * self._n_classes,
                "adaptive_branches": adaptive_branches.sum(-1).mean(),
                "method": method_name,
                "batch": batch,
            }
            if cycles_data.get(method_name, None) is not None:
                # Steps : {"Cycles": {}, "Exit Tree"}
                diz_cicli = cycles_data[method_name][str(batch)]
                exit_tree = {int(k): v["Exit Tree"] for k, v in diz_cicli.items()}
                cycles = {int(k): v["Cycles"] for k, v in diz_cicli.items()}
                result["adaptive_exit_tree"] = np.vectorize(exit_tree.get)(
                    stopping_trees
                ).mean()
                result["adaptive_cycles"] = np.vectorize(cycles.get)(
                    stopping_trees
                ).mean()
            result.update(scores)
            results.append(result)
        return results

    def adaptive_benchmark(
        self,
        X,
        y,
        bitwidth,
        thresholds,
        batches=(1, 2, 4, 8),
        methods=("score_margin", "agg_score_margin", "max", "agg_max"),
        cycles_data=None,
    ):
        results = list()
        logits = self.predict_probs(X, bitwidth=bitwidth)
        complexities = self.predict_complexity(X=X).swapaxes(1, 2)
        if "score_margin" in methods:
            sm = compute_score_margin(logits)
        max = compute_max_score(logits)
        if self.n_classes == 2:
            # Vicino a 0 o 2**bits vuol dire massima sicurezza per una classe
            max = np.abs(max)
        else:
            max += 2 ** (bitwidth - 1)

        for batch in batches:
            print("Batch ", batch)
            if (self.n_estimators / batch) <= 1:
                print("Batch", batch, "has no sense with ", self.n_estimators)
                continue
            to_take = [x for x in range((batch - 1), (self.n_estimators), (batch))]
            if self.n_estimators % batch != 0:
                to_take.append(self.n_estimators - 1)

            cum_logits = np.cumsum(logits, axis=0)
            cum_complexities = np.cumsum(complexities, axis=0)
            cum_logits = cum_logits[to_take]
            cum_complexities = cum_complexities[to_take]
            batch_logits = np.copy(logits)[to_take, :, :]

            if "agg_score_margin" in methods:
                agg_sm = compute_score_margin(cum_logits)
            agg_max = compute_max_score(cum_logits)
            if self.n_classes == 2:
                agg_max = np.abs(agg_max)
            else:
                agg_max += 2 ** (bitwidth - 1)

            # Agg SM
            if "agg_score_margin" in methods:
                results += self.compute_adaptive_method(
                    method_name="aggregated-score-margin",
                    logits=cum_logits,
                    y=y,
                    complexities=cum_complexities,
                    early_stop_scores=agg_sm,
                    batch=batch,
                    cycles_data=cycles_data,
                )

            if "agg_max" in methods:
                results += self.compute_adaptive_method(
                    method_name="aggregated-max",
                    logits=cum_logits,
                    y=y,
                    complexities=cum_complexities,
                    early_stop_scores=agg_max,
                    batch=batch,
                    cycles_data=cycles_data,
                )

            if "max" in methods:
                results += self.compute_adaptive_method(
                    method_name="max",
                    logits=batch_logits,
                    y=y,
                    complexities=cum_complexities,
                    early_stop_scores=max,
                    batch=batch,
                    cycles_data=cycles_data,
                )

            if "score_margin" in methods:
                results += self.compute_adaptive_method(
                    method_name="score-margin",
                    logits=batch_logits,
                    y=y,
                    complexities=cum_complexities,
                    early_stop_scores=sm,
                    batch=batch,
                    cycles_data=cycles_data,
                )

        return pd.DataFrame(results)

    def compute_qwyc(
        self,
        X_val,
        X_test,
        y_test,
        bitwidth,
        batches=(1, 2, 4, 8),
        cycles_data=None,
    ):
        results = list()
        for a in np.logspace(-4, -2, 10):
            qw = QWYC(
                base_model=self,
                alpha=a,
                tolerance=1,
                leaves_bits=bitwidth,
            )
            qw.fit_no_ordering(X_val)

            qw_ord = QWYC(
                base_model=self,
                alpha=a,
                tolerance=1,
                leaves_bits=bitwidth,
            )
            qw_ord.fit(X_val)
            for batch in batches:
                # QWYC unordered
                adaptive_predict, stopping_trees, adaptive_branches = qw.predict(
                    X_test,
                    batch=batch,
                )
                scores = score(y_test, adaptive_predict)
                scores = {"adaptive_" + k: v for k, v in scores.items()}
                result = {
                    "thr": a,
                    "adaptive_n_estimators": stopping_trees.mean(),
                    "adaptive_n_trees": stopping_trees.mean() * self._n_classes,
                    "adaptive_branches": adaptive_branches.mean(),
                    "method": "qwyc",
                    "batch": batch,
                }
                if cycles_data is not None:
                    # Steps : {"Cycles": {}, "Exit Tree"}
                    diz_cicli = cycles_data["qwyc"][str(batch)]
                    exit_tree = {int(k): v["Exit Tree"] for k, v in diz_cicli.items()}
                    cycles = {int(k): v["Cycles"] for k, v in diz_cicli.items()}
                    exit_trees = np.vectorize(exit_tree.get)(stopping_trees)
                    result["adaptive_exit_tree"] = exit_trees.mean()
                    result["adaptive_cycles"] = np.vectorize(cycles.get)(
                        stopping_trees
                    ).mean()
                result.update(scores)
                results.append(result)
                # QWYC ordered, batch = batch,
                adaptive_predict, stopping_trees, adaptive_branches = qw_ord.predict(
                    X_test, batch=batch
                )
                scores = score(y_test, adaptive_predict)
                scores = {"adaptive_" + k: v for k, v in scores.items()}
                result = {
                    "thr": a,
                    "adaptive_n_estimators": stopping_trees.mean(),
                    "adaptive_n_trees": stopping_trees.mean() * self._n_classes,
                    "adaptive_branches": adaptive_branches.mean(),
                    "method": "qwyc-ordered",
                    "batch": batch,
                }
                if cycles_data is not None:
                    # Steps : {"Cycles": {}, "Exit Tree"}
                    diz_cicli = cycles_data["qwyc"][str(batch)]
                    exit_tree = {int(k): v["Exit Tree"] for k, v in diz_cicli.items()}
                    cycles = {int(k): v["Cycles"] for k, v in diz_cicli.items()}
                    result["adaptive_exit_tree"] = np.vectorize(exit_tree.get)(
                        stopping_trees
                    ).mean()
                    result["adaptive_cycles"] = np.vectorize(cycles.get)(
                        stopping_trees
                    ).mean()
                result.update(scores)
                results.append(result)
        return pd.DataFrame(results)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import GradientBoostingClassifier
    from eden.utils import score

    ES = 7

    X, y = make_classification(
        n_features=20, n_classes=3, n_clusters_per_class=4, n_informative=6
    )
    b = GradientBoostingClassifier(
        random_state=0, max_depth=5, n_estimators=ES, init="zero"
    )
    b.fit(X, y)
    g = GBDT(base_model=b, bits_input=8, n_estimators=ES)

    bO = g.predict_complexity(X)
    bO = b.decision_function(X)
    gO = g.predict_probs(X).sum(axis=0).reshape(-1, 3)
    print(bO.shape)
    print(gO.shape)
    assert np.array_equal(bO, gO)

    gO = g.predict(X)
    bO = b.predict(X)
    print(gO)
    print(bO)
    assert np.array_equal(bO, gO)
    print(score(y, bO))
    print(score(y, gO))

    print(g.adaptive_benchmark(X, y, bitwidth=8, thresholds=np.arange(100)))
