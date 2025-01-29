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

        # Variables modified by the quantization functions, set to None by default
        self.alpha_scale = None
        self.alpha_zero_point = None
        self.leaf_scale = None
        self.leaf_zero_point = None

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
        return next(self.flat_trees[0].leaves).values.shape[0]

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
            for idx, tree in enumerate(self.flat_trees):
                values = [leaf.values for leaf in tree.leaves]
                if idx > 0:
                    min_max[0, 0] = min(min(values), min_max[0, 0])
                else:
                    min_max[0, 0] = min(values)
                min_max[0, 1] += max(values)
        min_val, max_val = min_max[:, 0].min(), min_max[:, 1].max()
        return min_val, max_val

    def predict_raw(self, X) -> np.ndarray:
        """
        Recursive prediction function for the ensemble.

        Parameters
        ----------
        X : np.ndarray
            The input data
        Returns
        -------
        np.ndarray
            The predictions with shape (n_samples, n_trees, leaf_shape)
        """
        predictions = np.zeros(
            (X.shape[0], self.n_trees, self.leaf_length),
            dtype=next(self.flat_trees[0].leaves).values.dtype,
        )

        for idx, tree in enumerate(self.flat_trees):
            for i_idx, x_in in enumerate(X):
                predictions[i_idx, idx] = tree.predict(x_in)
        return predictions
    
    def predict(self, X) -> np.ndarray:
        """ 
        Recursive prediction function for the ensemble, with aggregation.
        The function used to aggregated is self.aggregate_function
        Parameters
        ----------
        X : np.ndarray
            The input data
        Returns
        -------
        np.ndarray
            The predictions with shape (n_samples, output_length)
        """

        predictions = self.predict_raw(X)
        return self.aggreagate_function(predictions)

    def get_memory_cost(self, data_structure="arrays"):
        assert data_structure in ["arrays", "struct"]
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
        elif self.task in [
            "classification_multiclass_ovo",
            "regression",
            "classification_multiclass",
        ]:
            memory_cost["output"] = self.output_length * alphas.dtype.itemsize
        elif self.task == "classification_label":
            memory_cost["output"] = self.output_length * children_right.dtype.itemsize

        if data_structure == "struct":
            memory_cost["nodes"] = (
                memory_cost["children_right"]
                + memory_cost["features"]
                + memory_cost["alphas"]
            )
            memory_cost.pop("children_right"), memory_cost.pop(
                "features"
            ), memory_cost.pop("alphas")
        return memory_cost

    def get_access_cost(self, data_structure="arrays"):
        assert data_structure in ["arrays", "struct"]
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

        if data_structure == "struct":
            access_cost["nodes"] = (
                access_cost["children_right"]
                + access_cost["features"]
                + access_cost["alphas"]
            )
            access_cost.pop("children_right"), access_cost.pop(
                "features"
            ), access_cost.pop("alphas")
        return access_cost
    
    def remove_trees(self, idx : Union[int, List[int]]):
        """
        Prune one or more trees from the ensemble

        Parameters
        ----------
        idx : Union[int, List[int]]
            The index or list of indices to remove

        Returns
        -------
        Ensemble
            The ensemble without the trees
        """
        if isinstance(idx, int):
            idx = [idx]
        for i in sorted(idx, reverse=True):
            self.trees.pop(i)
        return self

    def __str__(self) -> str:
        return f"Ensemble of {self.n_trees} trees, input length {self.input_length}, output length {self.output_length}"
