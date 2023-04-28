from eden._types import _info_ctype
from typing import Dict
from collections import defaultdict
import logging


def _ensemble_memory_snapshot(
    *,
    ensemble_structure_mode: str,
    leaf_store_mode: str,
    n_leaves: int,
    n_features: int,
    n_trees: int,
    n_nodes: int,
    leaf_shape: int,
    output_shape: int,
    root_ctype: str,
    input_ctype: str,
    feature_idx_ctype: str,
    threshold_ctype: str,
    right_child_ctype: str,
    leaf_ctype: str,
) -> Dict[str, int]:
    memory_summary: Dict[str, int] = {}
    _, threshold_bits = _info_ctype(ctype=threshold_ctype)
    _, feature_idx_bits = _info_ctype(ctype=feature_idx_ctype)
    _, right_child_bits = _info_ctype(ctype=right_child_ctype)
    node_bits = threshold_bits + feature_idx_bits + right_child_bits

    _, input_bits = _info_ctype(ctype=input_ctype)
    _, root_bits = _info_ctype(ctype=root_ctype)
    _, leaf_bits = _info_ctype(ctype=leaf_ctype)

    memory_summary["INPUT"] = input_bits * n_features
    memory_summary["OUTPUT"] = leaf_bits * output_shape
    memory_summary["ROOTS"] = root_bits * n_trees
    if ensemble_structure_mode == "struct":
        # Simulate the struct padding, WIP
        node_bits = ((node_bits + 15) // 16) * 16
        memory_summary["NODES"] = node_bits * n_nodes
    else:
        memory_summary["FEATURE_IDX"] = feature_idx_bits * n_nodes
        memory_summary["THRESHOLDS"] = threshold_bits * n_nodes
        memory_summary["RIGHT_CHILDREN"] = right_child_bits * n_nodes
    if leaf_store_mode:
        memory_summary["LEAVES"] = leaf_bits * n_leaves * leaf_shape
    return memory_summary


def _compute_memory_map(*, target_architecture: str, ensemble_memory: Dict[str, int]):
    mapping = defaultdict(lambda: "")
    if target_architecture == "gap8":
        L1 = "PI_CL_L1"
        L2 = "PI_CL_L2"
        L1_SIZE = 64 * 100
        L2_SIZE = 512 * 100
    elif target_architecture == "pulpissimo":
        L1 = ""
        L2 = ""
        L1_SIZE = 0 * 100
        L2 = 520 * 100
    else:
        return mapping
    current_l1 = 0
    current_l2 = 0
    for key, val in ensemble_memory.items():
        if (current_l1 + val) < L1_SIZE:
            mapping[key] = L1
            current_l1 += val
        else:
            mapping[key] = L2
            current_l2 += val
        # TODO Extend for L3?
    if current_l2 > L2_SIZE:
        logging.warning("The ensemble may be too large")

    return mapping
