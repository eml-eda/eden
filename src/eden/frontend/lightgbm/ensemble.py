from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor
from typing import List, Optional
from eden.model.node import Node
from eden.model.ensemble import Ensemble
import numpy as np
from bigtree import find, print_tree
from eden.frontend.lightgbm.tree import parse_tree
from eden.model.aggregation import sum, mean


def parse_random_forest(*, model: LGBMModel, aggregate_function=sum):
    assert isinstance(
        model, (LGBMClassifier, LGBMRegressor)
    ), "Model must be a LGBMClassifier or LGBMRegressor"
    assert model.boosting_type in ["rf"], "Model must be a Random Forest"
    assert hasattr(model, "booster_"), "Model must be fitted"
    task = (
        "classification_multiclass"
        if isinstance(model, LGBMClassifier)
        else "regression"
    )
    column_names = model.feature_name_
    output_length: int = model.n_classes_ if isinstance(model, LGBMClassifier) else 1
    input_length: int = model.n_features_
    booster = model.booster_.trees_to_dataframe()
    trees: List[Node] = list()
    for tree_idx in booster.tree_index.unique():
        tree_df = booster[booster.tree_index == tree_idx]
        tree = parse_tree(
            tree_df=tree_df, column_names=column_names, input_length=input_length
        )
        trees.append(tree)

    ensemble = Ensemble(
        trees=trees,
        task=task,
        input_length=input_length,
        output_length=output_length,
        aggregate_function=aggregate_function,
    )
    return ensemble


# TODO: Can we avoid the scaling_by_lr? Maybe embed it in the aggregate function?
# It is still preferrable to scaling the values in the C code, it is just less efficient
def parse_boosting_trees(
    *, model: LGBMModel, aggregate_function=sum, scaling_by_lr=False
) -> Ensemble:
    """
    Parse an LGBMModel into an Ensemble object, works only if "gbdt" boosting type.
    Note that if scaling_by_lr is True, no sorting or adaptive should be done.

    Parameters
    ----------
    model : LGBMModel
        The fitted model to parse
    aggregate_function : _type_, optional
        Aggregation function, by default mean
    scaling_by_lr : bool, optional
        Statically scale leaf values by the LR, by default True

    Returns
    -------
    Ensemble
        the ensemble object
    """
    assert isinstance(
        model, (LGBMClassifier, LGBMRegressor)
    ), "Model must be a LGBMClassifier or LGBMRegressor"
    assert model.boosting_type in ["gbdt"], "Model must be a Boosting Tree (gbdt)"
    assert hasattr(model, "booster_"), "Model must be fitted"
    task = (
        # TODO: This part is not complete, what if it is OvO?
        "classification_multiclass"
        if isinstance(model, LGBMClassifier)
        else "regression"
    )
    learning_rate = model.learning_rate
    column_names = model.feature_name_
    output_length: int = model.n_classes_ if isinstance(model, LGBMClassifier) else 1
    input_length: int = model.n_features_
    booster = model.booster_.trees_to_dataframe()
    trees: List[Node] = list()
    for tree_idx in booster.tree_index.unique():
        tree_df = booster[booster.tree_index == tree_idx]
        # if scaling_by_lr and tree_idx > 0:
        #    tree_df.loc[:, "value"] *= learning_rate
        tree = parse_tree(
            tree_df=tree_df, column_names=column_names, input_length=input_length
        )
        trees.append(tree)

    ensemble = Ensemble(
        trees=trees,
        task=task,
        input_length=input_length,
        output_length=output_length,
        aggregate_function=aggregate_function,
    )
    return ensemble


if __name__ == "__main__":
    from lightgbm import LGBMRegressor
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = LGBMRegressor(
        boosting_type="gbdt",
        n_estimators=2,
        max_depth=3,
        learning_rate=0.1,
        num_leaves=31,
        subsample_freq=1,
        subsample=0.5,
        colsample_bytree=0.7,
        n_jobs=10,
        random_state=123456,
        verbose=-1,
    )
    model.fit(X, y)

    ensemble = parse_boosting_trees(model=model)
    print(ensemble)
    print_tree(ensemble.trees[0], attr_list=["alpha", "feature", "values"])
