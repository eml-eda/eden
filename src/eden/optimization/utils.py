from typing import Tuple
import math


def _get_bits_to_represent(
    *,
    range_val: Tuple[int, int],
    return_cvalid: bool = True,
) -> int:
    range_val = range_val[1] - range_val[0]
    n_bits = math.ceil(math.log2(range_val))
    if return_cvalid:
        # Set to multiple of 8
        n_bits = ((n_bits + 7) // 8) * 8
        if n_bits == 24:
            n_bits = 32
        elif n_bits > 32:
            raise NotImplementedError("Ensemble is too large")
    return n_bits
