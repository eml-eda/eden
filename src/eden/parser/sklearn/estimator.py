from typing import Union, MutableMapping, Mapping, List, Iterable
from sklearn import ensemble
from sklearn.ensemble._gb import BaseGradientBoosting
from sklearn.ensemble._forest import BaseForest
from sklearn import base
import eden
from .tree import parse_tree
from sklearn.tree import BaseDecisionTree


def _parse_ensemble(
    *,
    estimator: Union[BaseGradientBoosting, BaseForest],
) -> List[Mapping]:
    # Matrix of ensemble for GBT - Multiclass
    # list of ensembles for Rfs
    tree_list = list()
    # A single tree
    if not hasattr(estimator, "estimators_"):
        tree_dict = parse_tree(estimator=estimator)
        return [tree_dict]
    # An ensemble
    for tree_or_list in estimator.estimators_:
        if isinstance(tree_or_list, Iterable):
            for tree in tree_or_list:
                tree_dict = parse_tree(estimator=tree)
                tree_list.append(tree_dict)
        else:
            tree = tree_or_list
            tree_dict = parse_tree(estimator=tree)
            tree_list.append(tree_dict)

    return tree_list


def parse_estimator(
    *,
    estimator: Union[BaseGradientBoosting, BaseForest, BaseDecisionTree],
):
    ensemble_dictionary: MutableMapping = {}
    ensemble_dictionary["trees"] = _parse_ensemble(estimator=estimator)
    ensemble_dictionary["version"] = eden.__version__
    ensemble_dictionary["is_classification"] = isinstance(
        estimator, base.ClassifierMixin
    )
    ensemble_dictionary["n_estimators"] = (
        int(estimator.n_estimators) if hasattr(estimator, "n_estimators") else 1
    )

    ensemble_dictionary["is_forest"] = isinstance(estimator, BaseForest)
    ensemble_dictionary["n_trees"] = len(ensemble_dictionary["trees"])

    ensemble_dictionary["estimator"] = type(estimator).__name__
    ensemble_dictionary["input_len"] = int(estimator.n_features_in_)
    ensemble_dictionary["max_depth"] = max(
        [tree["max_depth"] for tree in ensemble_dictionary["trees"]]
    )
    ensemble_dictionary["n_nodes"] = sum(
        [tree["n_nodes"] for tree in ensemble_dictionary["trees"]]
    )
    ensemble_dictionary["n_leaves"] = sum(
        [tree["n_leaves"] for tree in ensemble_dictionary["trees"]]
    )
    ensemble_dictionary["eden_leaf_indicator"] = ensemble_dictionary["trees"][0][
        "eden_leaf_indicator"
    ]

    # Set this to one if we have two classes
    ensemble_dictionary["leaf_len"] = ensemble_dictionary["trees"][0]["leaf_len"]
    ensemble_dictionary["output_len"] = (
        int(estimator.n_classes_) if hasattr(estimator, "n_classes_") else 1
    )
    if (
        ensemble_dictionary["output_len"] == 2
        and ensemble_dictionary["is_classification"]
    ):
        ensemble_dictionary["output_len"] = 1
    return ensemble_dictionary
