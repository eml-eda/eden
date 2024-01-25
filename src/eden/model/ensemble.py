from eden.model.node import Node
from eden.export.carrays import ensemble_to_c_arrays
from typing import List, Union, Any, Callable
from bigtree import preorder_iter
from bigtree import print_tree
import numpy as np


class Ensemble:
    def __init__(
        self,
        *,
        trees: Union[List[Node], List[List[Node]]],
        task: str,
        input_length: int,
        output_length: int,
        aggregate_function: Callable,
    ) -> None:
        assert task in [
            "classification_multiclass",  # Binary or multiclass trees
            "classification_multiclass_ovo",  # One vs One multiclass/binary trees
            "classification_label",  # Trees with labels in leaves
            "regression",
        ]  # Regression ensembles
        self.task = task
        self.trees = trees
        self.input_length = input_length
        self.output_length = output_length
        self.aggreagate_function = aggregate_function

    # Always check the root node of the first tree,
    #  leaves still have unquantized values for thresholds
    @property
    def leaf_precision(self):
        return self.flat_trees[0].values[0].dtype

    @property
    def alpha_precision(self):
        return self.trees[0].alpha.dtype

    @property
    def leaf_length(self):
        return self.flat_trees[0].leaves[0].values.shape[0]

    @property
    def max_depth(self):
        return max([tree.max_depth for tree in self.flat_trees])

    @property
    def n_nodes(self):
        nodes = 0
        for tree in self.trees:
            nodes += len(list(preorder_iter(tree)))
        return nodes

    @property
    def n_leaves(self):
        leaves = 0
        for tree in self.trees:
            leaves += len(list(tree.leaves))
        return leaves

    @property
    def flat_trees(self) -> List[Node]:
        if all(isinstance(x, list) for x in self.trees):
            flat_trees = [tree for row in self.trees for tree in row]
        else:
            flat_trees = self.trees
        return flat_trees

    @property
    def n_trees(self) -> int:
        return len(self.flat_trees)
    
    @property
    def n_estimators(self) -> int:
        return len(self.trees)

    @property
    def leaf_range(self):
        # OVO case
        if self.task == "classification_multiclass_ovo":
            min_max = np.zeros((self.output_length, 2))
            for estimator in self.trees:
                for class_idx, tree in enumerate(estimator):
                    values = [leaf.values for leaf in tree.leaves]
                    min_max[class_idx, 0] += min(values)
                    min_max[class_idx, 1] += max(values)

        # Trees with multiclass
        elif self.task == "classification_multiclass":
            min_max = np.zeros((self.output_length, 2))
            min_max[:, 1] = self.n_trees
        # Regression case
        else:
            min_max = np.zeros((self.output_length, 2))
            # For RFs or models with probabilities, the min is 0 and the max is N_trees
            for estimator in self.trees:
                for tree in estimator:
                    values = [leaf.values for leaf in tree.leaves]
                    min_max[0, 0] += min(values)
                    min_max[0, 1] += max(values)
        min_val, max_val = min_max[:, 0].min(), min_max[:, 1].max()
        return min_val, max_val

    def predict(self, X, n_cores=1) -> np.ndarray:
        """
        Recursive prediction function for the ensemble.

        Parameters
        ----------
        X : np.ndarray
            The input data
        n_cores : int, optional
            Cores to be used, currently disabled, by default 1

        Returns
        -------
        np.ndarray
            The predictions with shape (n_samples, n_trees, n_classes)
        """
        predictions = np.zeros((X.shape[0], self.n_trees, self.output_length))

        for idx, tree in enumerate(self.flat_trees):
            for i_idx, x_in in enumerate(X):
                predictions[i_idx, idx] = tree.predict(x_in)
        # if self.aggregate_function is not None:
        #    self.aggreagate_function(predictions)
        return predictions

    def get_memory_cost(self):
        children_right, features, alphas, leaves, roots = ensemble_to_c_arrays(
            ensemble=self
        )
        memory_cost = {
            "input": self.input_length * alphas.dtype.itemsize,
            "children_right": children_right.nbytes,
            "features": features.nbytes,
            "roots": roots.nbytes,
            "alphas": alphas.nbytes,
        }
        if leaves is not None:
            memory_cost["leaves"] = leaves.nbytes
            memory_cost["output"] = self.output_length * leaves.dtype.itemsize
        elif self.task in ["classification_multiclass_ovo", "regression", "classification_multiclass"]:
            memory_cost["output"] = self.output_length * alphas.dtype.itemsize
        elif self.task == "classification_label":
            memory_cost["output"] = self.output_length * children_right.dtype.itemsize
        return memory_cost

    def get_access_cost(self):
        children_right, features, alphas, leaves, roots = ensemble_to_c_arrays(
            ensemble=self
        )
        access_cost = {
            "input": self.max_depth * self.n_trees,
            "children_right": self.max_depth * self.n_trees,
            "features": self.max_depth * self.n_trees,
            "roots": self.n_trees,
            "alphas": self.max_depth * self.n_trees,
        }
        if leaves is not None:
            access_cost["leaves"] = self.n_trees
        access_cost["output"] = self.n_trees * self.output_length
        return access_cost

    def __str__(self) -> str:
        return f"Ensemble of {self.n_trees} trees, input length {self.input_length}, output length {self.output_length}"
