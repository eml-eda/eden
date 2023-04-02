import numpy as np
from eden.utils import quantize
from eden.tree import Tree


class Ensemble:
    def __init__(self):
        pass

    def reorder_estimators(self, order):
        self.base_model.estimators[: self.n_estimators] = self.base_model.estimators_[
            : self.n_estimators
        ][order]

    def export_to_dict(self):
        represent = {
            "bits_inputs": self.bits_input,
            "bits_thresholds": self.bits_thresholds,
            "bits_feature_idx": self.bits_feature_idx,
            "bits_leaves_idx": self.bits_leaves_idx,
            "bits_nodes_idx": self.bits_nodes_idx,
            "bits_right_child": self.bits_right_child,
            "n_estimators": self.n_estimators,
            "n_trees": self.n_trees,
            "n_nodes": self.n_nodes,
            "n_leaves": self.n_leaves,
            "n_classes": self.n_classes,
            "n_features": self.n_features,
            "depth": self.depth,
        }

        return represent

    def get_structure(self, bits_leaves):
        # Quantize leaves
        struct_foglie = np.copy(self.struct_foglie)
        quant_struct_foglie = quantize(
            struct_foglie, range=self.range_foglie, bitwidth=bits_leaves
        )
        quant_struct_foglie = quant_struct_foglie.astype(int)

        # Quantize thresholds
        leaves_idx = self.struct_nodi[:, Tree.FEATURE_IDX] == -2
        quant_struct_nodi = np.copy(self.struct_nodi)
        quant_struct_nodi[leaves_idx, Tree.THRESHOLD] = np.floor(
            quant_struct_nodi[leaves_idx, Tree.THRESHOLD]
        )
        quant_struct_nodi = quant_struct_nodi.astype(int)
        return self.struct_radici, quant_struct_foglie, quant_struct_nodi
