import numpy as np
from jinja2 import Environment, FileSystemLoader
from glob import glob as glob
import shutil
from shutil import copytree as copy_tree
import os
from dataclasses import dataclass
from eden.utils import dtype_to_ctype, format_struct, format_array
from importlib_resources import files


class Ensemble2Template:
    DATA_PATH = files("ensembles").joinpath("data").joinpath("gap8")
    TEMPLATE_FOLDER_PATH = (
        files("ensembles").joinpath("data").joinpath("gap8").joinpath("templates")
    )
    DEPLOYMENT_PATH = "compile_autogen/"
    DEPLOYMENT_TEMPLATE_PATH = DEPLOYMENT_PATH + "autogen/"

    """
    Class to export all the common fields to a jinja2 template
    """

    def __init__(self, python_config):
        self.c_config = {}
        self.python_config = python_config
        # Convert dtypes
        self.c_config.update(self.extract_ensemble_constants(self.python_config))

    def extract_ensemble_constants(self, d):
        """
        d -> input dictionary
        """
        d_out = {}
        d_out["feature_dtype"] = dtype_to_ctype(bits=d["bits_inputs"], signed=True)
        # Include il -2 per le leaf, quindi signed
        d_out["feature_idx_dtype"] = dtype_to_ctype(
            bits=d["bits_feature_idx"], signed=True
        )
        d_out["node_idx_dtype"] = dtype_to_ctype(bits=d["bits_nodes_idx"], signed=False)
        d_out["node_shift_dtype"] = dtype_to_ctype(
            bits=d["bits_right_child"], signed=False
        )

        # Constants definitions
        d_out["n_estimators"] = d["n_estimators"]
        d_out["n_trees"] = d["n_trees"]
        d_out["n_classes"] = d["n_classes"] if d["n_classes"] > 2 else 1
        d_out["n_nodes"] = d["n_nodes"]
        d_out["n_features"] = d["n_features"]
        d_out["n_leaves"] = d["n_leaves"]

        # Da cambiare a seconda dell'ensemble
        has_leaves = (d_out["n_trees"] == d_out["n_estimators"]) and (
            d_out["n_classes"] > 2
        )
        # Gli indici delle foglie vengono sempre scritte nel campo threshold
        # Le foglie vengono scritte nel campo threshold se e' RF-binario o GBT
        if has_leaves:
            self.python_config["bits_thresholds_deployed"] = max(
                d["bits_leaves_idx"], d["bits_thresholds"]
            )
            self.python_config["bits_leaves_deployed"] = self.python_config[
                "bits_leaves"
            ]
        else:
            self.python_config["bits_thresholds_deployed"] = max(
                d["bits_leaves"], d["bits_thresholds"]
            )
            self.python_config["bits_leaves_deployed"] = self.python_config[
                "bits_thresholds_deployed"
            ]

        d_out["leaves_idx_dtype"] = dtype_to_ctype(
            bits=self.python_config["bits_thresholds_deployed"], signed=True
        )
        d_out["threshold_dtype"] = dtype_to_ctype(
            bits=self.python_config["bits_thresholds_deployed"], signed=True
        )
        d_out["leaves_dtype"] = dtype_to_ctype(
            bits=self.python_config["bits_leaves_deployed"], signed=True
        )
        d_out["logit_dtype"] = dtype_to_ctype(
            bits=self.python_config["bits_leaves_deployed"], signed=True
        )

        # Ragionamenti su L1/L2
        d_out["bits_l1_left"] = (
            44 * 8 * 1000
            - self.python_config["bits_inputs"] * self.python_config["n_features"]
            - d_out["n_classes"] * self.python_config["bits_leaves_deployed"]
        )
        assert d_out["bits_l1_left"] >= 0
        d_out["bits_nodes"] = (
            self.python_config["bits_thresholds_deployed"]
            + self.python_config["bits_feature_idx"]
            + self.python_config["bits_right_child"]
        ) * self.python_config["n_nodes"]
        return d_out

    def set_ensemble_data(self, roots, nodes, leaves=None):
        d_out = {}
        d_out["roots_data"] = format_array(roots)
        d_out["nodes_data"] = format_struct(nodes)
        if leaves is not None:
            d_out["leaves_data"] = format_array(leaves.reshape(-1))
        self.c_config.update(d_out)

    def set_dynamic_data(self, threshold, batch):
        d_out = {}
        d_out["threshold"] = threshold
        d_out["batch"] = batch
        self.c_config.update(d_out)

    def set_input_data(self, input_samples) -> dict:
        """
        Numpy sample to dict
        """
        data = {}
        data_inp = list()
        for inp in input_samples:
            input_sample = format_array(inp.reshape(-1))
            data_inp.append(input_sample)
        # Input sample
        data["input_samples"] = data_inp
        self.c_config.update(data)

    def write_ensemble(self):
        """
        Crea anche un file vuoto per gli inputs se non ci sono ancora.
        """
        assert "input_samples" in self.c_config, "No input specified"
        if os.path.exists(self.DEPLOYMENT_PATH):
            shutil.rmtree(self.DEPLOYMENT_PATH)
        os.makedirs(self.DEPLOYMENT_PATH)
        copy_tree(
            os.path.join(self.DATA_PATH, "src"),
            os.path.join(self.DEPLOYMENT_PATH, "src"),
        )
        copy_tree(
            os.path.join(self.DATA_PATH, "stats_lib/"),
            os.path.join(self.DEPLOYMENT_PATH, "stats_lib"),
        )
        copy_tree(
            os.path.join(
                self.DATA_PATH,
                f'include/{self.python_config["bits_leaves_deployed"]}bit',
            ),
            os.path.join(self.DEPLOYMENT_PATH, "include/"),
        )
        shutil.copy(
            os.path.join(self.DATA_PATH, f"include/ensemble.h"),
            os.path.join(self.DEPLOYMENT_PATH, "include/"),
        )

        templateLoader = FileSystemLoader(self.TEMPLATE_FOLDER_PATH)
        env = Environment(loader=templateLoader, trim_blocks=True, lstrip_blocks=True)

        os.makedirs(self.DEPLOYMENT_TEMPLATE_PATH)
        for template_path in glob(os.path.join(self.TEMPLATE_FOLDER_PATH, "*.jinja2")):
            template_bname = os.path.basename(template_path)
            template = env.get_template(template_bname)
            template = template.render(data=self.c_config)
            fname = template_bname.replace(".jinja2", "")
            if fname == "Makefile":
                target_folder = os.path.join(self.DEPLOYMENT_PATH, fname)
            else:
                target_folder = os.path.join(self.DEPLOYMENT_TEMPLATE_PATH, fname)
            with open(target_folder, "w") as f:
                f.write(template)

    def write_dynamic_config(self, batch):
        threshold = 2 ** (self.python_config["bits_leaves_deployed"] - 1)
        assert os.path.exists(os.path.join(self.DEPLOYMENT_PATH, "autogen"))
        self.set_dynamic_data(threshold, batch)

        templateLoader = FileSystemLoader(self.TEMPLATE_FOLDER_PATH)
        env = Environment(loader=templateLoader, trim_blocks=True, lstrip_blocks=True)

        template_name = "ensemble_dynamic_config.h.jinja2"
        template_path = os.path.join(self.TEMPLATE_FOLDER_PATH, template_name)
        template = env.get_template(template_name)
        template = template.render(data=self.c_config)
        fname = template_name.replace(".jinja2", "")
        with open(os.path.join(self.DEPLOYMENT_TEMPLATE_PATH, fname), "w") as f:
            f.write(template)
