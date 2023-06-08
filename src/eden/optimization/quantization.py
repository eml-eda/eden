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

from typing import Mapping, Optional, Tuple, Union, List
from copy import deepcopy
import numpy as np
from . import _DEPLOYMENT_KEY


def infer_dtype(*, quant_range: Tuple[int, int]) -> np.dtype:
    supported_dtype = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
    signed = quant_range[0] < 0
    for dt in supported_dtype:
        if np.iinfo(dt).min <= quant_range[0] and np.iinfo(dt).max >= quant_range[1]:
            return dt
    raise NotImplementedError(f"quant_range={quant_range} is not supported")


def _get_bits_boundaries(*, signed: bool, bits: int) -> Tuple[int, int]:
    if signed:
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1
    else:
        qmin = 0
        qmax = 2 ** (bits) - 1
    return qmin, qmax


def get_quantization_parameters(
    *,
    data_range: Tuple[float, float],
    bits: int,
    quantization_mode: str = "",
    signed: bool = False,
) -> Mapping[str, float]:
    qmin, qmax = _get_bits_boundaries(signed=signed, bits=bits)
    A, B = data_range
    S = (B - A) / (2**bits - 1)
    Z = -(round(A / S) - qmin)
    return {"S": S, "Z": Z}


def quantizer(
    *,
    data,
    output_bits: int,
    data_range: Optional[Tuple[float, float]] = None,
    quantization_mode: str = "uniform",
    signed: bool = False,
) -> Tuple[np.ndarray, Mapping[str, float]]:
    if quantization_mode != "uniform":
        raise NotImplementedError(
            f"quantization_mode = {quantization_mode} not supported"
        )
    if data_range is None:
        data_range = (data.min(), data.max())

    qmin, qmax = _get_bits_boundaries(signed=signed, bits=output_bits)

    quant_params: Mapping[str, float] = get_quantization_parameters(
        data_range=data_range,
        bits=output_bits,
        quantization_mode=quantization_mode,
        signed=signed,
    )
    data = np.copy(data)
    # TODO: Avoid truncating, but round instead. The sum of multiple tree logits leads
    # to overflow in C, truncating currently avoids it.
    data = np.trunc(data / quant_params["S"] + quant_params["Z"]).astype(int)
    data = np.clip(a=data, a_min=qmin, a_max=qmax)
    return data, quant_params


def quantize_estimator(
    *,
    quantization_aware_training: bool,
    estimator_dict: Mapping,
    input_range: Tuple[float, float],
    bits_input: Optional[int],
    bits_output: Optional[int],
) -> Union[Optional[Mapping], Optional[Mapping], Tuple[Mapping, Mapping]]:
    # No quantization specified
    if bits_input is None and bits_output is None:
        return estimator_dict
    quantized_ensemble = deepcopy(estimator_dict)
    quantized_trees = quantized_ensemble["trees"]

    input_qparams, output_qparams = None, None
    if bits_input is not None:
        # Extract quantization range
        quantized_trees, input_qparams = _quantize_ensemble_thresholds(
            quantization_aware_training=quantization_aware_training,
            trees_list=quantized_trees,
            bits=bits_input,
            input_range=input_range,
        )
        quantized_ensemble["input_qparams"] = input_qparams
    if bits_output is not None:
        quantized_trees, output_qparams = _quantize_ensemble_output(
            trees_list=quantized_trees, bits=bits_output
        )
        quantized_ensemble["output_qparams"] = output_qparams
    quantized_ensemble["trees"] = quantized_trees
    return quantized_ensemble


def _quantize_ensemble_thresholds(
    *,
    quantization_aware_training: bool,
    trees_list: List[Mapping],
    bits: int,
    input_range: Tuple[float, float],
):
    trees_list = deepcopy(trees_list)
    for idx, tree in enumerate(trees_list):
        if quantization_aware_training:
            qparams = None
            qdata = np.ceil(np.asarray(tree["threshold"])).astype(int)
        else:
            qdata, qparams = quantizer(
                data=np.asarray(tree["threshold"]),
                output_bits=bits,
                data_range=input_range,
            )
        tree["threshold"] = qdata.astype(int).tolist()
    return trees_list, qparams


def _quantize_ensemble_output(*, trees_list: List[Mapping], bits: int) -> List[Mapping]:
    # Get the ranges
    trees_list = deepcopy(trees_list)
    extrema = np.zeros((len(trees_list), trees_list[0]["leaf_len"], 2))
    for idx, tree in enumerate(trees_list):
        leaf_feature_value = tree["eden_leaf_indicator"]
        leaf_idx = np.asarray(tree["feature"]) == leaf_feature_value
        assert leaf_idx.sum() == tree["n_leaves"], "Invalid number of leaves"
        values = np.asarray(tree["value"])[leaf_idx]
        # Case 1: Classification OVA, Regression, Binary classification
        extrema[idx, :, 0] = np.min(values, axis=0)
        extrema[idx, :, 1] = np.max(values, axis=0)
    extrema = extrema.sum(axis=0)  # Simulate accumulation
    extrema = (np.min(extrema), np.max(extrema))
    print("Detected extrama", extrema)

    qparams = get_quantization_parameters(
        data_range=extrema, bits=bits, quantization_mode=""
    )
    # Actual quantization of the trees and reserialization
    for idx, tree in enumerate(trees_list):
        # TODO: Find a better way to implement this. Maybe set them to -1?
        # For now values != leaf are ignored, so they MUST be ignored in the
        # quantized dictionary
        qdata, _ = quantizer(
            data=np.asarray(tree["value"]), data_range=extrema, output_bits=bits
        )
        tree["value"] = qdata.astype(int).tolist()
    return trees_list, qparams
