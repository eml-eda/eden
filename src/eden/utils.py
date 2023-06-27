import math


def _compute_min_bits(min_val: float, max_val: float, return_cvalid: bool = True):
    range_val = max_val - min_val
    if range_val > 0:
        n_bits = math.ceil(math.log2(range_val))
    else:
        n_bits = 1
    if return_cvalid:
        # Set to multiple of 8
        n_bits = ((n_bits + 7) // 8) * 8
        if n_bits == 24:
            n_bits = 32
        elif n_bits > 32:
            raise NotImplementedError("Ensemble is too large")
    return n_bits


def _compute_ctype(min_val, max_val):
    n_bits = _compute_min_bits(min_val=min_val, max_val=max_val)
    base = "int"
    if min_val >= 0:
        base = "u" + base
    return f"{base}{int(n_bits)}_t"
