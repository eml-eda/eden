from typing import Any, Optional, List, Tuple, Dict, Union
import logging

import numpy as np
from eden._eden_ensemble import _EdenEnsemble
from eden._deployment import _deploy
from eden._parser import _parse_sklearn_model, _get_leaves_bounds
from eden._types import (
    _bits_to_represent,
    _round_up_pow2,
    _get_ctype,
    _ctypes_ensemble,
    _qtypes_ensemble,
    _merge_ctypes,
    _info_ctype,
)
from eden._quantization import _quantize_ensemble
from eden._memory import _ensemble_memory_snapshot, _compute_memory_map


def _infer_supported_simd(*, target_architecture: str):
    if target_architecture in ["gap8", "pulpissimo"]:
        return True
    else:
        return False


def _organize_data_fields(
    *,
    leaf_store_mode: str,
    threshold_ctype: str,
    leaf_ctype: str,
    right_child_ctype: str,
    n_trees: int,
    n_leaves: int,
    leaf_shape: int,
    feature_idx: np.ndarray,
    thresholds: np.ndarray,
    right_children: np.ndarray,
    leaves: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if leaf_store_mode == "auto":
        if leaf_shape != 1:
            leaf_store_mode = "external"
        else:
            leaf_store_mode = "internal"
    # Thresholds and Leaves are in the same struct field / array
    if leaf_store_mode == "internal":
        old_leaf_ctype = leaf_ctype
        merged_ctype = _merge_ctypes(ctypes=[leaf_ctype, threshold_ctype])
        leaf_ctype = merged_ctype
        threshold_ctype = merged_ctype

        for t in range(n_trees):
            thresholds[t][feature_idx[t] == -2] = leaves[t].reshape(-1)
            # TODO: There is probably a smarter way to solve this issue
            # Logits unsigned and threshold signed. If we quantize, what is the qtype?
            if leaf_ctype != "float":
                if "u" in old_leaf_ctype and "u" not in leaf_ctype:
                    logging.warning(
                        "Leaf were unsigned and thresholds signed. This may be bugged"
                    )
                    _, bits = _info_ctype(ctype=leaf_ctype)
                    thresholds[t][feature_idx[t] == -2] -= 2**bits - 1

    else:
        # Not necessary if we store the leaves in a matrix
        leaf_idx_ctype = _get_ctype(
            bits_word=_bits_to_represent(value=n_leaves), signed=False
        )
        merged_ctype = _merge_ctypes(ctypes=[right_child_ctype, leaf_idx_ctype])
        right_child_ctype = merged_ctype
        current_leaves = 0
        for t in range(n_trees):
            right_children[t][feature_idx[t] == -2] = np.arange(
                current_leaves, len(leaves[t]) + current_leaves
            )
            current_leaves = len(leaves[t]) + current_leaves

    return (
        leaf_store_mode,
        thresholds,
        threshold_ctype,
        right_children,
        right_child_ctype,
    )


def _infer_ensemble_structure(
    *,
    threshold_ctype: str,
    feature_idx_ctype: str,
    right_child_ctype: str,
) -> str:
    _, threshold_bits = _info_ctype(ctype=threshold_ctype)
    _, feature_idx_bits = _info_ctype(ctype=feature_idx_ctype)
    _, right_child_bits = _info_ctype(ctype=right_child_ctype)
    node_bits = threshold_bits + feature_idx_bits + right_child_bits
    # Structure mode
    padded_node_bits = min(_round_up_pow2(value=node_bits), 16)
    # TODO: Is this always true?
    if padded_node_bits > node_bits:
        ensemble_structure_mode = "array"
    else:
        ensemble_structure_mode = "struct"
    return ensemble_structure_mode


def convert(
    *,
    model: Any,
    # Quantization params
    test_data: Optional[np.ndarray] = None,
    ## Input parameters
    input_qbits: Optional[int] = None,  # Float default
    output_qbits: Optional[int] = None,  # Float default
    input_data_range: Optional[Tuple[float, float]] = None,
    # Changes what quantization we should perform
    quantization_aware_training: bool,
    # Deployment flags
    target_architecture: str = "any",
    leaf_store_mode: str = "auto",
    ensemble_structure_mode: str = "struct",
    use_simd: Union[bool, str] = "auto",
    output_folder: str = "eden-ensemble",
):
    # Model parsing
    (
        roots,
        feature_idx,
        thresholds,
        right_children,
        leaves,
        task,
        n_features,
        max_depth,
        n_estimators,
        n_trees,
        n_leaves,
        n_nodes,
        leaf_shape,
        output_shape,
    ) = _parse_sklearn_model(model=model)

    output_data_range = _get_leaves_bounds(
        n_trees=n_trees,
        task=task,
        n_estimators=n_estimators,
        leaves=leaves,
        leaf_shape=leaf_shape,
    )
    # C-style types
    (
        root_ctype,
        feature_idx_ctype,
        input_ctype,
        right_child_ctype,
        leaf_ctype,
    ) = _ctypes_ensemble(
        n_features=n_features,
        max_depth=max_depth,
        n_trees=n_trees,
        input_qbits=input_qbits,
        input_data_range=input_data_range,
        output_qbits=output_qbits,
        output_data_range=output_data_range,
    )
    # QTypes for quantized fields
    input_qtype, leaf_qtype = _qtypes_ensemble(
        input_qbits=input_qbits,
        input_data_range=input_data_range,
        output_qbits=output_qbits,
        output_data_range=output_data_range,
    )
    # Ensemble quantization
    (
        thresholds,
        leaves,
    ) = _quantize_ensemble(
        n_trees=n_trees,
        quantization_aware_training=quantization_aware_training,
        input_qtype=input_qtype,
        leaf_qtype=leaf_qtype,
        thresholds=thresholds,
        feature_idx=feature_idx,
        leaves=leaves,
    )

    # Optimizations based on the deployment flags
    ## Leaf store mode - threshold_ctype is created here, it may differ from input_ctype
    (
        leaf_store_mode,
        thresholds,
        threshold_ctype,
        right_children,
        right_child_ctype,  # It may be updated if we store larger indexes
    ) = _organize_data_fields(
        n_trees=n_trees,
        n_leaves=n_leaves,
        leaf_shape=leaf_shape,
        feature_idx=feature_idx,
        leaf_store_mode=leaf_store_mode,
        thresholds=thresholds,
        threshold_ctype=input_ctype,
        leaves=leaves,
        leaf_ctype=leaf_ctype,
        right_children=right_children,
        right_child_ctype=right_child_ctype,
    )
    # Memory occupation snapshot {FIELD: bits}
    if leaf_store_mode == "internal":
        n_leaves = 0
    # Struct or array?
    if ensemble_structure_mode == "auto":
        ensemble_structure_mode = _infer_ensemble_structure(
            threshold_ctype=threshold_ctype,
            feature_idx_ctype=feature_idx_ctype,
            right_child_ctype=right_child_ctype,
        )
    # TODO : The two dict(snapshot and map) could be merged?
    memory_snapshot: Dict[str, int] = _ensemble_memory_snapshot(
        ensemble_structure_mode=ensemble_structure_mode,
        leaf_store_mode=leaf_store_mode,
        n_leaves=n_leaves,
        n_features=n_features,
        n_trees=n_trees,
        n_nodes=n_nodes,
        leaf_shape=leaf_shape,
        output_shape=output_shape,
        root_ctype=root_ctype,
        input_ctype=input_ctype,
        feature_idx_ctype=feature_idx_ctype,
        threshold_ctype=threshold_ctype,
        right_child_ctype=right_child_ctype,
        leaf_ctype=leaf_ctype,
    )
    # Memory layer handling
    memory_map = _compute_memory_map(
        target_architecture=target_architecture,
        ensemble_memory=memory_snapshot,
    )
    # SIMD Enabler
    if use_simd == "auto":
        use_simd = _infer_supported_simd(target_architecture=target_architecture)
    if test_data is None:
        test_data = [np.zeros((n_features))]
        if input_qbits is not None:
            test_data = [inp.astype(np.int32) for inp in test_data]

    model = _EdenEnsemble(
        ROOTS=roots,
        FEATURE_IDX=feature_idx,
        THRESHOLDS=thresholds,
        RIGHT_CHILDREN=right_children,
        LEAVES=leaves,
        INPUT=test_data,
        task=task,
        n_estimators=n_estimators,
        n_trees=n_trees,
        n_nodes=n_nodes,
        n_leaves=n_leaves,
        n_features=n_features,
        leaf_shape=leaf_shape,
        output_shape=output_shape,
        input_data_range=input_data_range,
        output_data_range=output_data_range,
        root_ctype=root_ctype,
        input_ctype=input_ctype,
        feature_idx_ctype=feature_idx_ctype,
        threshold_ctype=threshold_ctype,
        right_child_ctype=right_child_ctype,
        leaf_ctype=leaf_ctype,
        input_qtype=input_qtype,
        threshold_qtype=input_qtype,
        leaf_qtype=leaf_qtype,
        target_architecture=target_architecture,
        ensemble_structure_mode=ensemble_structure_mode,
        leaf_store_mode=leaf_store_mode,
        use_simd=use_simd,
        memory_snapshot=memory_snapshot,
        memory_map=memory_map,
    )
    # Using a dataclass to represent the ensemble
    _deploy(output_folder=output_folder, ensemble=model)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    X, y = make_classification(random_state=0, n_classes=2, n_informative=10)
    clf = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=2)
    clf.fit(X, y)
    test_data = np.zeros(shape=(1, X.shape[1]))
    convert(
        model=clf,
        test_data=[test_data],
        input_qbits=32,
        output_qbits=32,
        input_data_range=(np.min(X), np.max(X)),
        quantization_aware_training=False,
    )
    predictions = np.asarray([t.predict_proba(test_data) for t in clf.estimators_]).sum(
        axis=0
    )
    print(predictions)
