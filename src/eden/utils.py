from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Iterable
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score
import joblib as jbl
import glob
import os


def infer_objects(df: pd.DataFrame):
    df_recast = df.copy()
    # print("Column conversion")
    for column in df.columns:
        if ((df[column] % 1) == 0).all():
            df_recast[column] = df[column].astype(np.int32)
        # print(f"Column {column}: dtype {df_recast[column].dtype}")
    return df_recast


def pareto(
    dataset, columnX: str, columnY: str, additional_columns=[], additional_orders=[]
):
    d = dataset.copy()
    d = d[~d[columnX].isna()]
    d = d[~d[columnY].isna()]
    for additional_column in additional_columns:
        d = d[~d[additional_column].isna()]
    d = d.sort_values(
        by=[columnX, columnY] + additional_columns,
        ascending=[True, False] + additional_orders,
    )
    d = d.drop_duplicates(subset=[columnX])
    d["local_maximum"] = d[columnY].expanding(axis=0).max()
    d = d[d[columnY] == d["local_maximum"]]
    d = d.drop("local_maximum", axis=1)
    d = d.drop_duplicates(subset=[columnY])
    return d


"""
Python -> C
"""


def dtype_to_ctype(bits: int, signed: bool):
    ctype = f"int{bits}_t"
    if not signed:
        ctype = "u" + ctype
    return ctype


def min_bits(max_int: int, has_negative_idx=False):
    """
    Return the minimum number of bits to represent the input
    """
    b = max_int + 1
    if has_negative_idx:
        b = b * 2
    n_bytes = int(np.ceil(np.log2(b) / 8))
    # Avoid 24 bits
    n_bytes = 4 if n_bytes == 3 else n_bytes
    bits = n_bytes * 8
    assert bits in [8, 16, 32]
    return bits


def format_array(array):
    arr = map(int, list(array))
    arr = map(str, arr)
    arr = ", ".join(arr)
    return arr


def format_struct(struct) -> str:
    c_struct = list()
    n_elem_riga = struct.shape[-1]
    struct = np.copy(struct).reshape(-1, n_elem_riga)
    struct = struct.astype(int, copy=True)

    for riga in struct:
        riga = list(riga)
        riga = ", ".join(map(str, riga))
        riga = "{%s}" % (riga)
        c_struct.append(riga)
    c_struct = ",\n".join(c_struct)
    return c_struct


"""
Quantizzazione
"""


def quantize(X, range, bitwidth, signed=True):
    if range[0] != 0:
        mas = max(abs(range[0]), abs(range[1]))
        mini = -max(abs(range[0]), abs(range[1]))
        range = (mini, mas)
    X_q = np.copy(X)
    min_q = -(2 ** (bitwidth - 1)) if signed else 0
    S = (range[1] - range[0]) / (2**bitwidth - 1)
    Z = -(range[0] / S - min_q)
    X_q = np.round(X_q / S + Z)
    assert X_q.max() <= (2 ** (bitwidth - 1))
    X_q = np.clip(X_q, -(2 ** (bitwidth - 1)), (2 ** (bitwidth - 1) - 1))
    return X_q


def score(y, y_hat):
    is_binary = len(np.unique(y)) == 2
    scores = dict()
    scores["accuracy"] = accuracy_score(y, y_hat)
    scores["balanced_accuracy"] = balanced_accuracy_score(y, y_hat)
    scores["f1"] = f1_score(y, y_hat, average="binary" if is_binary else "weighted")
    return scores


def list_to_str(lista: Iterable[int]) -> str:
    """
    Convert an input list with integers to a string for a csv
    """
    lista = deepcopy(lista)
    return "/".join(map(str, lista))


"""
Funzioni adaptive
"""


def compute_max_score(logits):
    """
    logits: an array with size [1,n_classes]
    """
    return np.max(logits, axis=-1)


def compute_score_margin(logits):
    """
    logits: an array with size [n_trees, n_samples ,n_classes]
    """
    local_logits = np.copy(logits)
    local_logits -= np.min(logits, axis=-1).reshape(logits.shape[0], logits.shape[1], 1)

    partial_sort_idx = np.argpartition(-logits, kth=1, axis=-1)
    partial_sort_val = np.take_along_axis(logits, partial_sort_idx, axis=-1)[:, :, :2]
    sm = np.abs(np.diff(partial_sort_val, axis=-1)).reshape(
        logits.shape[0], logits.shape[1]
    )
    return sm


def adaptive_predict(logits, branches, threshold, early_scores):
    """
    logits: array con le logit di piu' inputs, di dimensione
     [total_batches, samples, n_classes]
    threshold: early stopping threshold
    early_stop_metric: max o score margin
    """
    n_batches, n_samples, n_classes = logits.shape
    n_trees_per_estimator = branches.shape[-1]
    # Array storing the stopping tree of each sample
    classified_at_batch = np.zeros(shape=n_samples)
    adaptive_logits = np.zeros(shape=(n_samples, n_classes))
    adaptive_branches = np.zeros(shape=(n_samples, n_trees_per_estimator))
    # Compute the early stop metric for each level, BUT the last one
    for batch in range(n_batches - 1):
        # Compute the mask of samples stopping at this batch
        stopping_at_batch = early_scores[batch] > threshold
        # Avoid re-writing samples that stopped before:
        # Consider only entries with batch == 0
        stopping_at_batch &= classified_at_batch == 0
        # Update the classified mask
        classified_at_batch[stopping_at_batch] = batch + 1
        adaptive_logits[stopping_at_batch] = logits[batch, stopping_at_batch]
        adaptive_branches[stopping_at_batch] = branches[batch, stopping_at_batch]

    # All the yet unclassified instances are set equal to the final predictions
    assert classified_at_batch.max() <= (n_batches - 1), "Wrong number of estimators"
    stopping_at_batch = classified_at_batch == 0
    classified_at_batch[stopping_at_batch] = n_batches
    adaptive_logits[stopping_at_batch] = logits[-1, stopping_at_batch]
    adaptive_branches[stopping_at_batch] = branches[-1, stopping_at_batch]
    return adaptive_logits, adaptive_branches, classified_at_batch


def find_and_load(
    classifier: str,
    n_estimators: int,
    bits_input: int,
    dataset: str,
    max_depth: int = None,
    patient: int = None,
    temporal=None,
):
    """
    Load a joblib model from the log folder
    """
    pz = "" if patient is None else f"-patient{int(patient)}"
    tmp = ""
    if temporal is not None:
        tmp = "-temporal" if temporal else f"-notemporal"

    bi = f"-bitsinput{bits_input}"
    md = f"-maxdepth{max_depth}" if max_depth is not None else ""
    os.makedirs(f"logs/{classifier}/{dataset}", exist_ok=True)
    path = f"logs/{classifier}/{dataset}/{dataset}{pz}{md}{bi}-estimators*{tmp}.jbl"
    for model in glob.glob(path):
        estimators = int(
            model.split("estimators")[1].replace(".jbl", "").replace(tmp, "")
        )
        if estimators >= n_estimators:
            print("Loaded model ", model)
            clf = jbl.load(model)
            return clf
    return None
