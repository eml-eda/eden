from typing import Tuple, Optional, List
import numpy as np
from eden._types import _get_ctype, _get_qtype, _info_qtype, _range_of_qtype


def quantize(data: np.ndarray, qtype: str) -> np.ndarray:
    # TODO Check that QTYPE is valid
    _, _, bits_frac = _info_qtype(qtype=qtype)
    lower, upper = _range_of_qtype(qtype=qtype)
    data = np.copy(data)
    # Shift
    data = data * (2**bits_frac)
    # Round and cast to int
    data = np.round(data).astype(np.int32)
    # Saturate
    data[data > upper] = upper
    data[data < lower] = lower
    return data


def _quantize_ensemble(
    *,
    n_trees: int,
    quantization_aware_training: bool,
    input_qtype: Optional[int],
    leaf_qtype: Optional[str],
    thresholds: List[np.ndarray],
    feature_idx: List[np.ndarray],
    leaves: List[np.ndarray],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if input_qtype is not None:
        # QAT : Thresholds are already in fixed point
        if quantization_aware_training == True:
            for t in range(n_trees):
                thresholds[t][feature_idx[t] == -2] = np.ceil(
                    thresholds[t][feature_idx[t] == -2]
                )
        else:
            # Quantization : Fxp
            for t in range(n_trees):
                thresholds[t] = quantize(
                    data=thresholds[t],
                    qtype=input_qtype,
                )

    if leaf_qtype is not None:
        # Quantization : Fxp
        for t in range(n_trees):
            leaves[t] = quantize(
                data=leaves[t],
                qtype=leaf_qtype,
            )
    return (thresholds, leaves)
