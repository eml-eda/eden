from typing import Optional, Tuple, Union, List
import math


def suggest_qtype(bits: int, data_range: Tuple[float, float]) -> str:
    """
    Suggest a valid qtype that avoids overflow for the inputs

    Parameters
    ----------
    bits : int
        maximum bits to be used
    data_range : Tuple[float, float]
        range of the data to be quantized, in the form of (min, max)

    Returns
    -------
    str
        a valid qtype for eden
    """
    qtype = _get_qtype(bits_word=bits, data_range=data_range)
    return qtype


def _round_up_pow2(*, x: Union[int, float]) -> int:
    return int(math.pow(2, math.ceil(math.log(x) / math.log(2))))


def _bits_to_represent(*, value: int) -> int:
    return math.ceil(math.log2(value))


def _range_of_ctype(ctype: str) -> Tuple[int, int]:
    bits_ctype = int("".join([str(c) for c in ctype if c.isdigit()]))
    signed = "u" not in ctype
    minimum = 0 if not signed else -(2 ** (bits_ctype - 1))
    maximum = 2 ** (bits_ctype) - 1 if not signed else 2 ** (bits_ctype - 1) - 1
    return minimum, maximum


def _range_of_qtype(*, qtype: str) -> Tuple[int, int]:
    signed, bits_int, bits_frac = _info_qtype(qtype=qtype)
    bits_dtype = int(signed) + bits_int + bits_frac
    upper = (2**bits_dtype) - 1 if not signed else (2 ** (bits_dtype - 1) - 1)
    lower = 0 if not signed else -(2 ** (bits_dtype - 1))
    return lower, upper


def _merge_ctypes(*, ctypes: List[str]) -> str:
    if "float" in ctypes:
        return "float"
    v_min, v_max = _range_of_ctype(ctype=ctypes[0])
    for ctype in ctypes:
        c_min, c_max = _range_of_ctype(ctype=ctype)
        if v_min > c_min:
            v_min = c_min
        if v_max < c_max:
            v_max = c_max
    merged_bits = _bits_to_represent(value=max(abs(v_max), abs(v_min))) + int(v_min < 0)
    merged_ctype = _get_ctype(bits_word=merged_bits, signed=v_min < 0)
    return merged_ctype


def _info_qtype(*, qtype: str) -> Tuple[bool, int, int]:
    part_int, part_digits = qtype.split(".")
    part_int = int(part_int[1:])
    part_digits = int(part_digits)
    signed_qtype: bool = ((part_digits + part_int) % 2) != 0
    return signed_qtype, part_int, part_digits


def _info_ctype(*, ctype: str) -> Tuple[bool, int]:
    if ctype == "float":
        signed = True
        bits = 32
        return signed, bits
    signed = "u" not in ctype
    bits = int("".join([str(c) for c in ctype if c.isdigit()]))
    return signed, bits


def _get_qtype(*, bits_word: int, data_range: Tuple[float, float]) -> Optional[str]:
    if bits_word is None:
        return None
    lower_bound, upper_bound = data_range
    signed = lower_bound < 0
    bits_available: int = bits_word - int(signed)
    lower_bound: int = math.floor(lower_bound)
    upper_bound: int = math.ceil(upper_bound)

    bound = max(abs(lower_bound), abs(upper_bound))

    # Minimum number of bits to represent the integer part
    bits_int = math.ceil(math.log2(bound + 1))
    assert (
        bits_available >= bits_int
    ), f"Overflow: too few bits to represent range {data_range}"
    bits_frac = bits_available - bits_int
    qtype = f"Q{bits_int}.{bits_frac}"
    return qtype


def _get_ctype(*, bits_word: Optional[int], signed: bool):
    if bits_word is None:
        return "float"
    cbits = max(_round_up_pow2(x=bits_word), 8)
    ctype = f'{"u" if not signed else ""}int{int(cbits)}_t'
    return ctype


def _ctypes_ensemble(
    *,
    n_features: int,
    max_depth: int,
    n_trees: int,
    input_qbits: Optional[int],
    input_data_range: Tuple[float, float],
    output_qbits: Optional[int],
    output_data_range: Tuple[float, float],
):
    root_ctype = _get_ctype(
        bits_word=_bits_to_represent(value=2**n_trees), signed=False
    )
    # Include the -2
    feature_idx_ctype = _get_ctype(
        bits_word=_bits_to_represent(value=n_features) + 1, signed=True
    )
    input_ctype = _get_ctype(
        bits_word=input_qbits,
        signed=(input_data_range[0] < 0),  # and (input_qbits is None),
    )

    right_child_ctype = _get_ctype(
        bits_word=_bits_to_represent(value=2**max_depth), signed=False
    )
    leaf_ctype = _get_ctype(
        bits_word=output_qbits,
        signed=(output_data_range[0] < 0),  # and (output_qbits is None),
    )
    return root_ctype, feature_idx_ctype, input_ctype, right_child_ctype, leaf_ctype
