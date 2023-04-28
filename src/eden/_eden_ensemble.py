from dataclasses import dataclass
from typing import Optional, Mapping, List, Tuple
import numpy as np


@dataclass(frozen=True)
class _EdenEnsemble:
    # Data - Already quantized
    ## Ensemble Data
    ## TODO Understand if it should be stored elsewhere
    ROOTS: List[np.ndarray]
    FEATURE_IDX: List[np.ndarray]
    THRESHOLDS: List[np.ndarray]
    RIGHT_CHILDREN: List[np.ndarray]
    LEAVES: List[np.ndarray]
    ## Test data
    INPUT: List[np.ndarray]

    # ensemble info
    # These could also be computed in a post-init
    task: str
    n_estimators: int
    n_trees: int
    n_nodes: int
    n_leaves: int
    n_features: int
    leaf_shape: int
    output_shape: int
    input_data_range: Tuple[float, float]
    output_data_range: Tuple[float, float]

    # ctypes
    root_ctype: str
    input_ctype: str
    feature_idx_ctype: str
    threshold_ctype: str
    right_child_ctype: str
    leaf_ctype: str
    # qtypes
    input_qtype: Optional[str]
    threshold_qtype: Optional[str]
    leaf_qtype: Optional[str]

    # configs
    target_architecture: str
    ensemble_structure_mode: str
    leaf_store_mode: str
    use_simd: bool

    # memory
    memory_snapshot: Mapping[str, int]
    memory_map: Mapping[str, str]

    def __repr__(self) -> str:
        pass
