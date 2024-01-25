import time
from eden.export.arrays import tree_to_arrays
import numpy as np

if __name__ == "__main__":
    # This is a benchmark on how fast are the predictions
    # Less efficient than the recursive version, unused for now
    def predict(
        X: np.ndarray,
        children_left: np.ndarray,
        children_right: np.ndarray,
        alphas: np.ndarray,
        features: np.ndarray,
        values: np.ndarray,
    ):
        output = np.zeros((X.shape[0], values.shape[-1]))
        stop = X.shape[-1]
        for input_idx in range(X.shape[0]):
            node_idx = 0
            while features[node_idx] != stop:
                if X[input_idx, :][features[node_idx]] <= alphas[node_idx]:
                    node_idx = children_left[node_idx]
                else:
                    node_idx = children_right[node_idx]
                output[input_idx] = values[node_idx]

        return output

    def run_ensemble(model, ensemble, X):
        X1 = np.zeros((X.shape[0], ensemble.n_trees, ensemble.output_length))
        X2 = np.zeros((X.shape[0], ensemble.n_trees, ensemble.output_length))
        X3 = np.zeros((X.shape[0], ensemble.n_trees, ensemble.output_length))
        start = time.time()
        for idx, tree in enumerate(model.estimators_):
            X1[:, idx, :] = tree.predict_proba(iris.data)
        stop = time.time()
        print("Time to run ensemble (sklearn): ", stop - start, "seconds")

        start = time.time()
        X2 = ensemble.predict(X)
        stop = time.time()
        print("Time to run ensemble (recursion): ", stop - start, "seconds")

        start = time.time()
        for idx, tree in enumerate(ensemble.flat_trees):
            children_left, children_right, features, alphas, values = tree_to_arrays(
                tree=tree
            )
            X3[:, idx, :] = predict(
                X, children_left, children_right, alphas, features, values
            )
        stop = time.time()
        print("Time to run ensemble (iterative): ", stop - start, "seconds")
        assert np.array_equal(X1, X3)

    from sklearn.datasets import load_iris, fetch_california_housing
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from eden.frontend.sklearn import parse_random_forest
    from eden.model.ensemble import Ensemble

    iris = load_iris()
    model = RandomForestClassifier(n_estimators=100, random_state=0, max_depth=15)
    model.fit(iris.data, iris.target)

    model2 = parse_random_forest(model=model)

    run_ensemble(model, model2, iris.data)
