from lightgbm import LGBMModel, LGBMClassifier, LGBMRegressor
from typing import List, Optional
from eden.model.node import Node
from eden.model.ensemble import Ensemble
import numpy as np
from bigtree import find, print_tree
from eden.frontend.lightgbm.tree import parse_tree
from eden.model.aggregation import sum


def parse_random_forest(*, model : LGBMModel, aggregate_function = sum):
    assert isinstance(model, (LGBMClassifier, LGBMRegressor)), "Model must be a LGBMClassifier or LGBMRegressor"
    assert hasattr(model, "booster_"), "Model must be fitted"
    task  = "classification_multiclass" if isinstance(model, LGBMClassifier) else "regression"
    column_names = model.feature_name_

    model._n_classes
    output_length: int = (
        model.n_classes_ if isinstance(model, LGBMClassifier) else 1
    )
    input_length: int = model.n_features_
    booster = model.booster_.trees_to_dataframe()
    trees : List[Node] = list()
    for tree_idx in booster.tree_index.unique():
        tree_df = booster[booster.tree_index == tree_idx]
        tree = parse_tree(tree_df = tree_df, column_names = column_names, input_length=input_length)
        trees.append(tree)
                
    ensemble = Ensemble(trees= trees, task = task, input_length=input_length, output_length=output_length, aggregate_function=aggregate_function)
    return ensemble






    


if __name__=="__main__":
    from lightgbm import LGBMRegressor
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True)
    model = LGBMRegressor(
        boosting_type="rf",
        n_estimators=10,
        max_depth=4,
        learning_rate=0.1,
        num_leaves = 31,
        subsample_freq = 1,
        subsample = 0.5,
        colsample_bytree = 0.7,
        n_jobs = 10,
        random_state = 123456

    )
    model.fit(X, y)

    ensemble = parse_random_forest(model = model)
    print(ensemble)
    print_tree(ensemble.trees[0], attr_list=["alpha", "feature" , "values"])
