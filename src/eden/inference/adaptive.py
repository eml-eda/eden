# *--------------------------------------------------------------------------*
# * Copyright (c) 2023 Politecnico di Torino, Italy                          *
# * SPDX-License-Identifier: Apache-2.0                                      *
# *                                                                          *
# * Licensed under the Apache License, Version 2.0 (the "License");          *
# * you may not use this file except in compliance with the License.         *
# * You may obtain a copy of the License at                                  *
# *                                                                          *
# * http://www.apache.org/licenses/LICENSE-2.0                               *
# *                                                                          *
# * Unless required by applicable law or agreed to in writing, software      *
# * distributed under the License is distributed on an "AS IS" BASIS,        *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
# * See the License for the specific language governing permissions and      *
# * limitations under the License.                                           *
# *                                                                          *
# * Author: Francesco Daghero francesco.daghero@polito.it                    *
# *--------------------------------------------------------------------------*

import numpy as np


def predict_adaptive(
    predictions: np.ndarray, thresholds: np.ndarray, early_scores: np.ndarray
):
    """
    Compute the final predictions for each threshold using the early stopping scores
      provided

    Parameters
    ----------
    predictions : np.ndarray
        The estimator output (raw logits), shape must be {n_classifiers, n_samples, n_classes}
        See Notes for more details.
    thresholds : np.ndarray
        Array of thresholds to benchmark, shape {n_thresholds}
    early_scores : np.ndarray
        Array of early stopping scores, shape {n_classifiers, n_samples}

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Predictions and classifiers used.
    Notes
    -----
    No accumulation is performed for aggregated metrics. It should be done beforehand in
    a similar way:
    predictions = np.cumsum(predictions, axis = 0)
    """
    assert len(predictions.shape) == 3, "If you have 2 classes, reshape the array"
    assert early_scores.shape[0] == predictions.shape[0]
    assert early_scores.shape[1] == predictions.shape[1]
    n_models, n_samples, n_classes = predictions.shape
    assert n_classes >= 2
    n_thresholds = thresholds.shape[0]

    # Predictions
    # adaptive_predictions_mask = np.zeros((n_thresholds, n_models, n_samples))
    adaptive_predictions = np.zeros((n_thresholds, n_samples, n_classes))
    # Stopping estimator
    classifiers_used = np.zeros(shape=(n_thresholds, n_samples), dtype=int)
    # Mask to keep track of not classified (faster, but expensive)
    unclassified_mask = np.ones(shape=(n_thresholds, n_samples), dtype=bool)

    for classifier_idx in range(n_models - 1):
        stopped_samples = np.greater(
            early_scores[classifier_idx], thresholds[:, None]
        )  # mask [n_thresholds, n_samples]
        # Exclude the samples already stopped, faster than removing elements dynamically
        actual_stopped_samples = unclassified_mask & stopped_samples
        classifiers_used[actual_stopped_samples] = classifier_idx + 1
        unclassified_mask &= ~actual_stopped_samples
    # Final classifier, label everything unlabeled
    classifiers_used[unclassified_mask] = n_models
    # adaptive_predictions[i]
    # broad_logits = np.broadcast_to(
    #    logits[:, None, :, :], (n_models, n_thresholds, n_samples, n_classes)
    # )
    # l = np.take(broad_logits, classifiers_used[:, :] - 1, axis=0)
    adaptive_predictions = np.take_along_axis(
        predictions, indices=classifiers_used[:, :, None] - 1, axis=0
    )
    return adaptive_predictions, classifiers_used