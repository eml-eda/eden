from bigtree import preorder_iter
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
import numpy as np
import logging


def rf_to_onnx(*, ensemble):
    logging.warning("This function is still experimental")
    assert ensemble.task in ["classification_multiclass"]
    # Extract some variables before conversion
    task = ensemble.task
    input_name = "inputs"

    # Onnx fields
    class_ids = list()
    class_nodeids = list()
    class_treeids = list()
    # Class probabilities / n_trees
    class_weights = list()
    classlabels_int64s = [*range(ensemble.output_length)]
    nodes_falsenodeids = list()
    nodes_featureids = list()
    nodes_modes = list()
    nodes_nodeids = list()
    nodes_treeids = list()
    nodes_truenodeids = list()
    nodes_values = list()
    post_transform = "NONE"

    # Extract the data, per tree
    # Index are reset to 0 for each tree
    for tree_idx, tree in enumerate(ensemble.flat_trees):
        preorder_nodes = [*preorder_iter(tree)]
        for node_idx, node in enumerate(preorder_nodes):
            if node.is_leaf:
                class_ids.extend([*range(ensemble.output_length)])
                class_nodeids.extend([node_idx for _ in range(ensemble.output_length)])
                class_treeids.extend([tree_idx for _ in range(ensemble.output_length)])
                class_weights.extend(list(node.values.reshape(-1)))
                # classlabels_int64
                nodes_modes.append("LEAF")
                nodes_values.append(0)
                nodes_featureids.append(0)
            else:
                nodes_modes.append("BRANCH_LEQ")
                nodes_values.append(node.alpha)
                nodes_featureids.append(node.feature)
            nodes_treeids.append(tree_idx)
            nodes_nodeids.append(node_idx)

            if node.right is not None:
                nodes_falsenodeids.append(preorder_nodes.index(node.right))
            else:
                nodes_falsenodeids.append(0)

            if node.left is not None:
                nodes_truenodeids.append(preorder_nodes.index(node.left))
            else:
                nodes_truenodeids.append(0)

    assert (
        len(class_nodeids) == len(class_treeids) == len(class_weights)
    ), "Missed some leaves"
    assert (
        len(nodes_falsenodeids)
        == len(nodes_featureids)
        == len(nodes_modes)
        == len(nodes_nodeids)
        == len(nodes_treeids)
        == len(nodes_truenodeids)
        == len(nodes_values)
    ), "Missed some nodes"

    nodesEnsemble = helper.make_node(
        op_type="TreeEnsembleClassifier",  # Use the TreeEnsembleClassifier operator
        name="TreeEnsembleClassifier",
        inputs=[input_name],
        outputs=["classes", "probs"],
        domain="ai.onnx.ml",
    )
    # This seems the only way to pass the parameters to the operator
    nodesEnsemble.attribute.extend(
        [
            helper.make_attribute("class_ids", class_ids, "INTS"),
            helper.make_attribute("class_nodeids", class_nodeids, "INTS"),
            helper.make_attribute("class_treeids", class_treeids, "INTS"),
            helper.make_attribute("class_weights", class_weights, "FLOATS"),
            helper.make_attribute("classlabels_int64s", classlabels_int64s, "INTS"),
            helper.make_attribute("nodes_falsenodeids", nodes_falsenodeids, "INTS"),
            helper.make_attribute("nodes_featureids", nodes_featureids, "INTS"),
            helper.make_attribute("nodes_nodeids", nodes_nodeids, "INTS"),
            helper.make_attribute("nodes_modes", nodes_modes, "STR"),
            helper.make_attribute("nodes_treeids", nodes_treeids, "INTS"),
            helper.make_attribute("nodes_truenodeids", nodes_truenodeids, "INTS"),
            helper.make_attribute("nodes_values", nodes_values, "FLOATS"),
            helper.make_attribute("post_transform", post_transform, "STRING"),
        ]
    )

    graph = helper.make_graph(
        nodes=[
            nodesEnsemble,
        ],
        name="RandomForestModel",
        inputs=[
            helper.make_tensor_value_info(
                input_name, TensorProto.FLOAT, [None, ensemble.input_length]
            )  # Adjust shape if needed
        ],
        outputs=[
            helper.make_tensor_value_info(
                output_name,
                TensorProto.INT64,
                [
                    None,
                ],
            ),  # Adjust shape if needed
            helper.make_tensor_value_info(
                "probs", TensorProto.FLOAT, [None, ensemble.output_length]
            ),  # Adjust shape if needed
        ],
    )
    model = helper.make_model(
        graph=graph,
        ir_version=9,
        opset_imports=[
            helper.make_opsetid("", 19),
            helper.make_opsetid("ai.onnx.ml", 1),
        ],
    )
    return model
