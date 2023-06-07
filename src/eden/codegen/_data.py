from typing import Mapping
import numpy as np
from ._formatter import to_c_array2d, to_c_array, to_node_struct
import os


def _write_template_data(lookup, output_dir, template_data_src: Mapping) -> Mapping:
    os.makedirs(os.path.join(output_dir, "autogen"), exist_ok=True)
    roots_string = None
    node_string = None
    feature_string = None
    children_right_string = None
    threshold_string = None

    root_list = [t["n_nodes"] for t in template_data_src["trees"]]
    root_list.pop(-1)
    root_list.insert(0, 0)
    root_list = np.cumsum(np.asarray(root_list)).astype(int).tolist()
    roots_string = to_c_array(root_list)
    feature_list = list()
    children_right_list = list()
    threshold_list = list()
    for idx, tree in enumerate(template_data_src["trees"]):
        feature = np.asarray(tree["feature"])
        feature_list.append(feature)
        threshold = np.asarray(tree["threshold"])
        threshold_list.append(threshold)
        children_right = np.asarray(tree["children_right"])
        children_right_list.append(children_right)

    # feature_list = np.asarray(feature_list)
    # threshold_list = np.asarray(threshold_list)
    # children_right_list = np.asarray(children_right_list)
    node_string = to_node_struct(feature_list, threshold_list, children_right_list)
    feature_string = to_c_array(feature_list)
    threshold_string = to_c_array(threshold_list)
    children_right_string = to_c_array(children_right_list)

    leaf_string = ""
    if template_data_src["leaves_store_mode"] == "external":
        leaf_list = list()
        for tree in template_data_src["trees"]:
            leaf_feature_value = tree["eden_leaf_indicator"]
            leaf_idx = np.asarray(tree["feature"]) == leaf_feature_value
            assert leaf_idx.sum() == tree["n_leaves"], "Invalid number of leaves"
            values = np.asarray(tree["value"])[leaf_idx]
            leaf_list.append(values)
        leaf_string = to_c_array2d(leaf_list)

    output_fname = os.path.join(output_dir, "autogen", "eden_ensemble_data.h")
    template = lookup.get_template("eden_ensemble_data.h")
    t = template.render(
        roots_string=roots_string,
        node_string=node_string,
        feature_string=feature_string,
        threshold_string=threshold_string,
        children_right_string=children_right_string,
        leaf_string=leaf_string,
    )

    with open(f"{output_fname}", "w") as out_file:
        out_file.write(t)
    file_extension = os.path.splitext(output_fname)[1]
    if file_extension in [".c", ".h"]:
        os.system(f"clang-format -i {output_fname}")
    return template_data_src
