from eden.model.ensemble import Ensemble
import shutil
import numpy as np
import os
from eden.backend.allocation import _cache_placer
from eden.export.carrays import ensemble_to_c_arrays
import pkg_resources
from mako.lookup import TemplateLookup
from importlib.resources import files
from eden.backend.utils import (
    nptype_to_ctype,
    ctype_to_vtype,
    to_c_array,
    to_c_array2d,
    to_node_struct,
)
from typing import Optional
import subprocess

from collections import defaultdict

TEMPLATE_DIR = files("eden").joinpath("backend", "codegen", "templates")
DATA_DIR = files("eden").joinpath("backend", "codegen", "include")



class Deployment:
    """
    Configuration of the deployed ensemble
    """

    def __init__(
        self,
        children_right,
        features,
        alphas,
        leaves,
        roots,
        target,
        task,
        n_estimators,
        memory_cost,
        access_cost,
        output_length,
        input_length,
        input_data,
    ):
        self.children_right = children_right
        self.features = features
        self.alphas = alphas
        self.leaves = leaves
        self.roots = roots
        self.input_data = input_data
        self.target = target
        self.task = task
        # Disabled atm
        self.data_structure = "array"
        assert self.data_structure in ["array", "struct"]

        # Extract the statistics of the ensemble
        self.n_trees = len(self.roots)
        self.n_nodes = len(self.features)
        self.n_estimators = n_estimators
        self.input_length = input_length
        if self.task == "classification_multiclass" and output_length == 2:
            self.output_length = 1
        self.leaf_length = leaves.shape[-1] if leaves is not None else 1

        self.bits_output = memory_cost["output"] * 8 / output_length
        self.bits_input = memory_cost["input"] * 8 / input_length
        self.bits_children_right = children_right.dtype.itemsize * 8
        self.bits_features = features.dtype.itemsize * 8
        self.bits_alphas = alphas.dtype.itemsize * 8
        self.bits_roots = roots.dtype.itemsize * 8
        self.bits_leaves = leaves.dtype.itemsize * 8 if leaves is not None else None

        # Extract the ctypes/vtypes
        if self.leaves is not None:
            self.output_ctype = nptype_to_ctype(self.leaves.dtype)
        elif self.task in ["classification_multiclass_ovo", "regression"]:
            self.output_ctype = nptype_to_ctype(self.alphas.dtype)
        else:
            self.output_ctype = nptype_to_ctype(self.children_right.dtype)

        self.output_vtype = ctype_to_vtype(self.output_ctype)
        self.child_right_ctype = nptype_to_ctype(self.children_right.dtype)
        self.alpha_ctype = nptype_to_ctype(self.alphas.dtype)
        self.feature_ctype = nptype_to_ctype(self.features.dtype)
        self.root_ctype = nptype_to_ctype(self.roots.dtype)
        self.input_ctype = nptype_to_ctype(self.alphas.dtype)

        self.buffer_allocation = defaultdict(lambda : "")
        # Select the ltype if necessary
        if target == "gap8":
            cost_dict = {}
            for k, v in memory_cost.items():
                cost_dict[k] = (v, access_cost[k])
            self.buffer_allocation.update(_cache_placer(**cost_dict))

    @property
    def features_str(self):
        return to_c_array(self.features)

    @property
    def alphas_str(self):
        return to_c_array(self.alphas)

    @property
    def children_right_str(self):
        return to_c_array(self.children_right)

    @property
    def roots_str(self):
        return to_c_array(self.roots)

    @property
    def nodes_str(self):
        return to_node_struct(
            self.roots, self.features, self.alphas, self.children_right
        )

    @property
    def leaves_str(self):
        return to_c_array2d(self.leaves)

    @property
    def input_str(self):
        return [to_c_array(i.reshape(-1), separator_string="") for i in self.input_data] 


def deploy_model(
    *, ensemble, output_path, target="default", input_data: Optional[np.ndarray] = None
):
    target = target.lower()
    assert target in ["gap8", "pulpissimo", "default"], "Target must be GAP8 or GAP9"
    # Get the model arrays
    children_right, features, alphas, leaves, roots = ensemble_to_c_arrays(
        ensemble=ensemble
    )

    # Get the memory cost of each array.
    memory_cost = ensemble.get_memory_cost()
    access_cost = ensemble.get_access_cost()

    # All the logic is inside this class
    config = Deployment(
        children_right=children_right,
        features=features,
        alphas=alphas,
        leaves=leaves,
        roots=roots,
        target=target,
        task=ensemble.task,
        n_estimators= ensemble.n_estimators,
        memory_cost=memory_cost,
        input_data=input_data,
        access_cost=access_cost,
        output_length=ensemble.output_length,
        input_length=ensemble.input_length,
    )

    # Write the data template.
    lookup = TemplateLookup(
        directories=[TEMPLATE_DIR.joinpath("src"), TEMPLATE_DIR.joinpath("include"), TEMPLATE_DIR],
        strict_undefined=True,
    )
    template = lookup.get_template(f"main.c.t")
    main_c = subprocess.check_output(
        ["clang-format", "-style=LLVM", "-assume-filename=code.c"],
        input=template.render(config=config),
        text=True,
    )
    # print(rendered)
    template = lookup.get_template(f"ensemble.h.t")
    ensemble_h = template.render(config=config)
    ensemble_h = subprocess.check_output(
        ["clang-format", "-style=LLVM", "-assume-filename=code.h"],
        input=ensemble_h,
        text=True,
    )
    template = lookup.get_template(f"ensemble_data.h.t")
    ensemble_data_h = template.render(config=config)
    ensemble_data_h = subprocess.check_output(
        ["clang-format", "-style=LLVM", "-assume-filename=code.h"],
        input=ensemble_data_h,
        text=True,
    )

    template = lookup.get_template(f"input.h.t")
    input_h = template.render(config=config)
    input_h = subprocess.check_output(
        ["clang-format", "-style=LLVM", "-assume-filename=code.h"],
        input=input_h,
        text=True,
    )
    #print(rendered)
    
    template = lookup.get_template(f"Makefile.t")
    makefile = template.render(config=config)
    # Write the main file


    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path,"src"), exist_ok=True)
    os.makedirs(os.path.join(output_path,"include"), exist_ok=True)

    with open(os.path.join(output_path,"src", "main.c"), "w") as f:
        f.write(main_c)
    with open(os.path.join(output_path,"include", "ensemble.h"), "w") as f:
        f.write(ensemble_h)
    with open(os.path.join(output_path,"include", "ensemble_data.h"), "w") as f:
        f.write(ensemble_data_h)
    with open(os.path.join(output_path,"include", "input.h"), "w") as f:
        f.write(input_h)
    with open(os.path.join(output_path, "Makefile"), "w") as f:
        f.write(makefile)
    
    # Copy the include files in the include folder depending on the target
    if os.path.exists(os.path.join(DATA_DIR.joinpath(target))):
        for file in os.listdir(DATA_DIR.joinpath(target)):
            shutil.copy2(src= os.path.join(DATA_DIR.joinpath(target), file), dst= os.path.join(output_path,'include', file))


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from eden.frontend.sklearn import parse_random_forest
    from eden.transform.quantization import (
        quantize_leaves,
        quantize_post_training_alphas,
        quantize_pre_training_alphas,
        quantize
    )

    iris = load_iris()
    iris_qdata = quantize(data = iris.data, precision=8, min_val= iris.data.min(), max_val=iris.data.max())
    iris.target = [0 if i == 0 else 1 for i in iris.target]
    model = RandomForestClassifier(
        n_estimators=10, random_state=0, max_depth=7, min_samples_leaf=10
    )

    model.fit(iris_qdata, iris.target)
    print(model.predict_proba(iris_qdata)[:10])
    mod = parse_random_forest(model=model)
    quantized = quantize_leaves(estimator=mod, precision=8)
    data_min, data_max = iris.data.min(), iris.data.max()
    #quantized =quantize_post_training_alphas(estimator=quantized, precision=8, min_val= data_min, max_val=data_max)
    quantized = quantize_pre_training_alphas(estimator=quantized, precision=8)

    deploy_model(ensemble=quantized, target="default", output_path="generated_tests", input_data=iris_qdata)
    print(quantized.predict(iris_qdata).sum(axis=1)[:10])
