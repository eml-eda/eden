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

import math
from copy import deepcopy
import numpy as np
from eden.optimization.quantization import quantize, quantize_alphas, get_output_range
from eden.parser.sklearn.estimator import parse_estimator_data, parse_estimator_info
from eden.utils import _compute_min_bits, _compute_ctype
from eden.optimization.leaf_placer import (
    _get_optimal_leaves_placement,
    _place_leaves,
    collapse_same_class_nodes,
)
from eden.codegen import export
from typing import Tuple, Optional
from eden.optimization.cache import _cache_placer


class EdenGarden:
    """
    Main class to quantize and export a fitted estimator from sklearn in C
    """

    def __init__(
        self,
        *,
        estimator,
        input_range: Tuple[float, float],
        input_qbits: Optional[int],
        output_qbits: Optional[int],
        quantization_aware_training: bool = False,
        store_class_in_leaves: bool = False,
    ):
        """
        Init method, designed as in sklearn. No logic happens here.

        Parameters
        ----------
        estimator :
            A fitted sklearn estimator (a decision tree or a tree ensemble)
        input_range : Tuple[float, float]
            Range of the input values, for alphas quantization
        input_qbits, output_qbits : Optional[int]
            Quantization bits, use None for no quantization
        quantization_aware_training : bool, optional
            True if the input was quantized before fit, by default False
        """
        self.estimator = estimator
        self.input_qbits = input_qbits
        self.input_range = input_range
        self.output_qbits = output_qbits
        self.quantization_aware_training = quantization_aware_training
        self.store_class_in_leaves = store_class_in_leaves

    def fit(self, X_test: Optional[np.ndarray] = None):
        """
        Main conversion step of the ensemble.


        Parameters
        ----------
        X_test : np.ndarray, optional
           Data to be written in a C header file, by default None
           If None, generates a single input with all 0s

        Returns
        -------
        self
        """
        # Modifies a copy of the estimator, values are kept in the arrays
        # Only entries are removed, works only for SKLEARN
        # TODO: Make this library-agnostic, or move it to parsers
        if self.store_class_in_leaves:
            self.estimator = collapse_same_class_nodes(clf=self.estimator)

        # Extract the needed statistics for deployment
        (
            self.n_estimators_,
            self.n_trees_,
            self.max_depth_,
            self.input_len_,
            self.n_nodes_,
            self.n_leaves_,
            self.leaf_len_,
            self.output_len_,
        ) = parse_estimator_info(estimator=self.estimator)
        # Fake input generation
        if X_test is None:
            self.X_test_ = np.zeros((1, self.input_len_))
        else:
            self.X_test_ = X_test
        # Some default bits, includes float
        self.input_bits_ = 32
        self.output_bits_ = 32

        # Quantization
        if self.input_qbits is not None:
            self.estimator = quantize_alphas(
                estimator=self.estimator,
                input_min_val=self.input_range[0],
                input_max_val=self.input_range[1],
                method="qat" if self.quantization_aware_training else "post-fit",
                bits=self.input_qbits,
            )
            self.X_test_ = quantize(
                data=self.X_test_,
                min_val=self.input_range[0],
                max_val=self.input_range[1],
                bits=self.input_qbits,
            )
            self.input_bits_ = self.input_qbits
            self.input_range_ = (0, 2**self.input_qbits)

        # Extract the data, already concatenated
        (
            self.root_,
            self.feature_,
            self.threshold_,
            self.children_right_,
            self.children_left_,
            self.leaf_value_,
        ) = parse_estimator_data(estimator=self.estimator)

        # Leaf preparation here
        if self.store_class_in_leaves:
            self.leaf_value_ = np.argmax(self.leaf_value_, axis=-1).reshape(-1, 1)
            self.output_bits_ = _compute_min_bits(0, self.leaf_value_.max())
            self.output_range_ = (0, self.leaf_value_.max())
            # Eden sees leaves as already quantized
            self.output_qbits_ = self.output_bits_
            self.leaf_len_ = 1
        # Output quantization
        elif self.output_qbits is not None:
            self.output_range_ = get_output_range(estimator=self.estimator)
            self.leaf_value_ = quantize(
                data=self.leaf_value_,
                min_val=self.output_range_[0],
                max_val=self.output_range_[1],
                bits=self.output_qbits,
            )
            self.output_bits_ = self.output_qbits
            self.output_range_ = (0, 2**self.output_qbits)

        return self

    def _prepare_deployment_structures(self):
        # Compute bits
        self.children_right_bits_ = _compute_min_bits(
            min_val=0, max_val=self.children_right_.max()
        )
        self.leaves_idx_bits_ = _compute_min_bits(min_val=0, max_val=self.n_leaves_)
        self.root_bits_ = _compute_min_bits(min_val=0, max_val=self.root_.max())
        self.feature_bits_ = _compute_min_bits(min_val=0, max_val=self.feature_.max())
        self.node_bits_ = (
            self.feature_bits_ + self.input_bits_ + self.children_right_bits_
        )

        # Leaf - External vs Internal
        if self.leaf_len_ > 1:
            self.c_leaf_data_store_ = "external"
        # Note that the leaves were already prepared in the fit
        elif self.store_class_in_leaves:
            self.c_leaf_data_store_ = "internal"
        else:
            self.c_leaf_data_store_ = _get_optimal_leaves_placement(
                bits_children_right=self.children_right_bits_,
                bits_thresholds=self.input_bits_,
                bits_output=self.output_bits_,
                bits_leaves_idx=self.leaves_idx_bits_,
                n_leaves=self.n_leaves_,
                n_nodes=self.n_nodes_,
            )
        self.children_right_, self.threshold_ = _place_leaves(
            mode=self.c_leaf_data_store_,
            root=self.root_,
            children_right=self.children_right_,
            leaf_value=self.leaf_value_,
            threshold=self.threshold_,
        )

    def _prepare_deployment_ctypes(self):
        self.root_ctype_ = _compute_ctype(min_val=0, max_val=self.root_.max())
        self.feature_ctype_ = _compute_ctype(
            min_val=self.feature_.min(), max_val=self.feature_.max()
        )
        self.children_right_ctype_ = _compute_ctype(
            min_val=0, max_val=self.children_right_.max()
        )
        # Input
        if self.input_qbits is not None:
            self.threshold_ctype_ = _compute_ctype(
                min_val=self.threshold_.min(), max_val=self.threshold_.max()
            )
            self.input_ctype_ = _compute_ctype(
                min_val=self.input_range_[0], max_val=self.input_range_[1]
            )
        else:
            self.threshold_ctype_ = "float"
            self.input_ctype_ = "float"
        if self.output_qbits is not None:
            self.output_ctype_ = _compute_ctype(
                min_val=self.output_range_[0], max_val=self.output_range_[1]
            )
        else:
            self.output_ctype_ = "float"

    def _prepare_deployment_cache(self):
        # Placer input
        root_buffer = (self.n_trees_ * self.root_bits_, self.n_trees_)
        node_buffer = (self.n_nodes_ * self.node_bits_, math.log2(self.n_nodes_))
        input_buffer = (self.n_nodes_ * self.node_bits_, self.n_nodes_)
        output_buffer = (self.n_nodes_ * self.node_bits_, self.n_nodes_)
        leaf_buffer = (self.n_leaves_ * self.output_bits_, math.log2(self.n_leaves_))
        feature_buffer = (
            self.n_nodes_ * self.feature_bits_,
            math.log2(self.n_nodes_),
        )
        children_right_buffer = (
            self.n_nodes_ * self.children_right_bits_,
            math.log2(self.n_nodes_),
        )
        threshold_buffer = (
            self.n_nodes_ * self.input_bits_,
            math.log2(self.n_nodes_),
        )

        if self.c_leaf_data_store_ == "internal":
            # Compatibility var for templates
            self.leaf_ltype_ = ""
            (
                self.root_ltype_,
                self.node_ltype_,
                self.input_ltype_,
                self.output_ltype_,
            ) = _cache_placer(
                root_buffer=root_buffer,
                node_buffer=node_buffer,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
            )

            (
                _,
                self.feature_ltype_,
                self.children_right_ltype_,
                self.threshold_ltype_,
                _,
                _,
            ) = _cache_placer(
                root_buffer=root_buffer,
                feature_buffer=feature_buffer,
                children_right_buffer=children_right_buffer,
                threshold_buffer=threshold_buffer,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
            )
        else:
            (
                self.root_ltype_,
                self.node_ltype_,
                self.input_ltype_,
                self.output_ltype_,
                self.leaf_ltype_,
            ) = _cache_placer(
                root_buffer=root_buffer,
                node_buffer=node_buffer,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                leaf_buffer=leaf_buffer,
            )

            (
                _,
                self.feature_ltype_,
                self.children_right_ltype_,
                self.threshold_ltype_,
                _,
                _,
                _,
            ) = _cache_placer(
                root_buffer=root_buffer,
                feature_buffer=feature_buffer,
                children_right_buffer=children_right_buffer,
                threshold_buffer=threshold_buffer,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                leaf_buffer=leaf_buffer,
            )

    def deploy(
        self, *, deployment_folder: str = "./eden-ensemble/", target: str = "all"
    ):
        """
        Exports the current ensemble.

        Parameters
        ----------
        deployment_folder : str
            Path to the output directory, by default "./eden-ensemble/"
        """
        assert hasattr(self, "n_estimators_"), "Call .fit() first"
        assert target in (
            "all",
            "gcc",
            "pulpissimo",
            "gap8",
        ), f"Invalid target {target}"
        self._prepare_deployment_structures()
        self._prepare_deployment_ctypes()
        self._prepare_deployment_cache()
        # Finally export to C
        self._dump_model(deployment_folder=deployment_folder, target=target)

    def _dump_model(self, *, deployment_folder: str, target: str):
        export(eden_model=self, deployment_folder=deployment_folder, target=target)

    def __repr__(self):
        return vars(self)
