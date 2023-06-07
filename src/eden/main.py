"""
Entry point of the package
"""
import eden
import numpy as np
from typing import Any, Optional, Mapping, Tuple, List
from eden.parser import sklearn as sk_parse
from eden.optimization.leaf_placer import (
    prepare_leaves_placement,
    merge_leaves_in_thresholds,
)
from eden.optimization.indexes import compute_bits_indexes
from eden.optimization.quantization import quantize_estimator
from eden.optimization.cache import cache_placer
from eden import codegen


def convert_to_eden(
    *,
    # Base model info
    estimator: Any,
    estimator_library: str = "scikit-learn",
    # Quantization
    quantization_aware_training: bool,
    input_qbits: Optional[int],
    input_data_range: Tuple[float, float],
    output_qbits: Optional[int],
    # Store mode
    leaves_store_mode: str = "auto",
    ensemble_structure_mode: str = "struct",
    # Test data
    test_data: Optional[List[np.ndarray]] = None,
    # Output
    output_dir: str = "eden-ensemble",
) -> Mapping:
    assert estimator_library in ["scikit-learn"]
    if estimator_library == "scikit-learn":
        estimator_dictionary = sk_parse.parse_estimator(estimator=estimator)

    # Detect the best store mode for leaves and prepare the right child
    # No changes on the trees
    estimator_dictionary, leaves_store_mode = prepare_leaves_placement(
        estimator_dict=estimator_dictionary,
        leaf_placement_strategy=leaves_store_mode,
        input_qbits=input_qbits,
        output_qbits=output_qbits,
    )
    estimator_dictionary["leaves_store_mode"] = leaves_store_mode
    estimator_dictionary = compute_bits_indexes(estimator_dict=estimator_dictionary)
    # Quantization
    estimator_dictionary = quantize_estimator(
        quantization_aware_training=quantization_aware_training,
        estimator_dict=estimator_dictionary,
        input_range=input_data_range,
        bits_input=input_qbits,
        bits_output=output_qbits,
    )

    # Merge leaves inside the right struct
    if leaves_store_mode == "internal":
        estimator_dictionary = merge_leaves_in_thresholds(
            estimator_dict=estimator_dictionary
        )
    # Compute the memory map
    estimator_dictionary = cache_placer(
        estimator_dict=estimator_dictionary,
        input_bits=input_qbits,
        output_bits=output_qbits,
    )
    # Quantize some inputs if provided.
    if test_data is None:
        test_data = [
            np.zeros(
                estimator_dictionary["input_len"],
                dtype=int if input_qbits is not None else float,
            )
        ]

    # C conversion
    ensemble_dictionary = codegen.export(
        estimator_dict=estimator_dictionary, output_dir=output_dir, test_data=test_data
    )


if __name__ == "__main__":
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris, load_diabetes

    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=16)
    model.fit(X, y)
    convert_to_eden(
        estimator=model,
        quantization_aware_training=False,
        input_qbits=8,
        input_data_range=(X.min(), X.max()),
        output_qbits=8,
        leaves_store_mode="auto",
        ensemble_structure_mode="auto",
    )
