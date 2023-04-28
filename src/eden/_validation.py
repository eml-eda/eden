# TBD
def _check_valid_ctype_and_qtype(*, ctype: str, qtype: str) -> None:
    signed_ctype, cbits = _get_data_ctype(ctype=ctype)
    signed_qtype, qbits_int, qbits_fract = _get_data_qtype(qtype=qtype)
    is_valid = ~(signed_ctype ^ signed_qtype)
    if not is_valid:
        raise NotImplementedError()
    is_valid = cbits == (qbits_fract + qbits_int)
    assert is_valid, ValueError(f"Mismatch between ctype={ctype} and qtype={qtype}")


def _check_valid_qtype(*, qtype: str):
    values: Tuple[int, ...] = (32, 31, 16, 15, 8, 7)
    regex: str = "^(q|Q)[0-9]{1,2}\.[0-9]{1,2}$"
    has_correct_format = re.match(pattern=regex, string=qtype) is not None
    if has_correct_format:
        part_int, part_digits = qtype.split(".")
        part_int = int(part_int[1:])
        part_digits = int(part_digits)
        has_valid_bits = (part_int + part_digits) in values
        has_correct_format = has_correct_format and has_valid_bits
    return has_correct_format


def _check_valid_ctype(*, ctype: str):
    regex: str = "^(u?int(8|16|32)_t|float|double)$"
    is_ctype = re.match(pattern=regex, string=ctype) is not None
    return is_ctype
