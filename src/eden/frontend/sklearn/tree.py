# *--------------------------------------------------------------------------*
# * Copyright (c) 2023 Politecnico di Torino, Italy                          *
# * SPDX-License-Identifier: Apache-2.0                                      *
# *                                                                          *
# * Licensed under the Apache License, Version 2.0 (the "License");          *
# * you may not use this file except in compliance with the License.         *
# * You may obtain a copy of the License at                                  *
# *                                                                          *
# * http://www.apache.org/licenses/LICENSE-2.0                               *
# *                                                                          *
# * Unless required by applicable law or agreed to in writing, software      *
# * distributed under the License is distributed on an "AS IS" BASIS,        *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
# * See the License for the specific language governing permissions and      *
# * limitations under the License.                                           *
# *                                                                          *
# * Author: Francesco Daghero francesco.daghero@polito.it                    *
# *--------------------------------------------------------------------------*


from sklearn.tree import BaseDecisionTree, DecisionTreeClassifier
from eden.model.node import Node
from bigtree import print_tree
import numpy as np
from sklearn.tree._tree import TREE_LEAF


def parse_tree(model: BaseDecisionTree) -> Node:
    def traverse(node_idx, model) -> Node:
        input_length: int = model.n_features_in_
        float_values = model.tree_.value[node_idx]
        if isinstance(model, DecisionTreeClassifier):
            float_values = (float_values / float_values.sum(-1)[:, None]).astype(
                np.float32
            )
        alpha = model.tree_.threshold[node_idx].astype(np.float32)
        if model.tree_.children_left[node_idx] == TREE_LEAF:  # leaf node
            return Node(
                name=node_idx,
                values_samples=model.tree_.value[node_idx],
                values=float_values,
                feature=input_length,
                alpha=alpha,
                input_length=input_length,
            )
        else:  # decision node
            left = traverse(model.tree_.children_left[node_idx], model)
            right = traverse(model.tree_.children_right[node_idx], model)
            return Node(
                name=node_idx,
                values_samples=model.tree_.value[node_idx],
                values=float_values,
                feature=model.tree_.feature[node_idx],
                alpha=alpha,
                left=left,
                right=right,
                input_length=input_length,
            )

    return traverse(0, model)  # start from the root node


if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer as load_dataset
    from sklearn.tree import DecisionTreeClassifier as DTModel

    iris = load_dataset()
    model = DTModel(max_depth=2, random_state=0)

    model.fit(iris.data, iris.target)
    new_model = parse_tree(model)
    print_tree(new_model, attr_list=["name", "values", "feature", "threshold"])

    print(new_model.predict(iris.data[-1]))
    print(model.predict(iris.data[-1].reshape(1, -1)))
