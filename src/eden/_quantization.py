from typing import Tuple, Optional, List, Dict
import numpy as np
from eden._types import _get_ctype, _get_qtype, _info_qtype, _range_of_qtype


def _get_qparams(
    qbits: int,
    data_range: Tuple[float, float],
    signed: bool = False,
    symmetric: bool = True,
):
    if symmetric:
        if signed:
            bound = max(abs(data_range[0]), abs(data_range[1]))
            data_range = (-bound, bound)
    qmin = 0
    qmax = 2 ** (qbits) - 1
    if signed:
        qmin = -(2 ** (qbits - 1))
        qmax = 2 ** (qbits - 1) - 1

    A, B = data_range
    s = (B - A) / (2**qbits - 1)
    if not symmetric:
        z = -(round(A / s) - qmin)
    else:
        z = 0
    qparams = {"s": s, "z": z, "bits": qbits, "signed": signed}
    return qparams


def quantize(
    data: np.ndarray,
    qbits: int,
    signed: bool = False,
    data_range: Optional[Tuple[float, float]] = None,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Fixed point quantization

    Parameters
    ----------
    data : np.ndarray
        Data to be quantized
    qbits : int
        Bits to be used for quantization
    signed: bool
        Controls if the qdata is signed or not.
    data_range : Optional[Tuple[float, float]], default
        Tuple with the range of the data in format (min, max). If None,
        use the `data` field to extract the range. Default is None

    Returns
    -------
    np.ndarray
       The quantized data
    """
    if data_range is None:
        data_range = (data.min(), data.max())
    if symmetric:
        if signed:
            bound = max(abs(data_range[0]), abs(data_range[1]))
            data_range = (-bound, bound)

    qmin = 0
    qmax = 2 ** (qbits) - 1
    if signed:
        qmin = -(2 ** (qbits - 1))
        qmax = 2 ** (qbits - 1) - 1

    A, B = data_range
    s = (B - A) / (2**qbits - 1)
    if not symmetric:
        z = -(round(A / s) - qmin)
    else:
        z = 0
    qparams = {"s": s, "z": z, "bits": qbits, "signed": signed}

    data = np.copy(data)
    data = np.round(data / s + z).astype(int)
    data = np.clip(a=data, a_min=qmin, a_max=qmax)
    return data


def dequantize(data: np.ndarray, qparams: Dict[str, float]) -> np.ndarray:
    data = np.copy(data)
    data = (data - qparams["z"]) * qparams["s"]
    return data


def _quantize_ensemble(
    *,
    n_trees: int,
    quantization_aware_training: bool,
    input_qbits: Optional[Dict[str, float]],
    input_data_range: Tuple[float, float],
    leaf_qbits: Optional[Dict[str, float]],
    output_data_range: Tuple[float, float],
    thresholds: List[np.ndarray],
    feature_idx: List[np.ndarray],
    leaves: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if input_qbits is not None:
        # QAT : Thresholds are already in fixed point
        if quantization_aware_training == True:
            for t in range(n_trees):
                thresholds[t][feature_idx[t] != -2] = np.ceil(
                    thresholds[t][feature_idx[t] != -2]
                )
                thresholds[t] = thresholds[t].astype(int)
        else:
            # Quantization : Fxp
            for t in range(n_trees):
                thresholds[t] = quantize(
                    data=thresholds[t],
                    qbits=input_qbits,
                    data_range=input_data_range,
                    signed=input_data_range[0] < 0,
                )

    if leaf_qbits is not None:
        for t in range(n_trees):
            leaves[t] = quantize(
                data=leaves[t],
                qbits=leaf_qbits,
                data_range=output_data_range,
                signed=output_data_range[0] < 0,
            )
    return (thresholds, leaves)


def _qparams_ensemble(
    input_qbits: Optional[int],
    input_data_range: Tuple[float, float],
    output_qbits: Optional[int],
    output_data_range: Tuple[float, float],
):
    input_qparams, output_qparams = None, None
    if input_qbits is not None:
        input_qparams = _get_qparams(
            qbits=input_qbits,
            data_range=input_data_range,
            signed=input_data_range[0] < 0,
        )
    if output_qbits is not None:
        output_qparams = _get_qparams(
            qbits=output_qbits,
            data_range=output_data_range,
            signed=output_data_range[0] < 0,
        )
    return input_qparams, output_qparams
