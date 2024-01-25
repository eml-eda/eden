from eden.transform.pruning import prune_same_class_leaves
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from eden.frontend.sklearn import parse_random_forest


def test_prune_same_class_leaves():
    iris = load_iris()
    model = RandomForestClassifier(
        n_estimators=1, random_state=0, max_depth=7, min_samples_leaf=10
    )

    model.fit(iris.data, iris.target)

    mod = parse_random_forest(model=model)
    pruned = prune_same_class_leaves(estimator=mod)

    at_least_one_to_prune_flag = False
    for t in mod.trees:
        for leaf in t.leaves:
            at_least_one_to_prune_flag = (
                leaf.values.argmax() == leaf.siblings[0].values.argmax()
            )
    assert at_least_one_to_prune_flag

    for t in pruned.trees:
        for leaf in t.leaves:
            assert leaf.values.argmax() != leaf.siblings[0].values.argmax()
