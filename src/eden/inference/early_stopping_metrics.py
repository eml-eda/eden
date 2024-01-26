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


def score_margin(predictions):
    predictions = np.copy(predictions)
    predictions -= np.min(predictions, axis=-1).reshape(
        predictions.shape[0], predictions.shape[1], 1
    )

    partial_sort_idx = np.argpartition(-predictions, kth=1, axis=-1)
    partial_sort_val = np.take_along_axis(predictions, partial_sort_idx, axis=-1)[
        :, :, :2
    ]
    sm = np.abs(np.diff(partial_sort_val, axis=-1)).reshape(
        predictions.shape[0], predictions.shape[1]
    )
    return sm


def aggregated_score_margin(predictions):
    predictions = np.copy(predictions)
    predictions = np.cumsum(predictions, axis=0)
    return score_margin(predictions=predictions)


def score_max(predictions):
    assert (
        len(predictions.shape[1]) == 3
    ), "Predictions should have shape [N_CLASSIFIERS, N_SAMPLES, N_CLASSES]"
    assert len(predictions.shape[-1]) == 1, "N_CLASSES should be minimum 2"
    predictions = np.copy(predictions)
    return np.max(predictions, axis=-1)


def aggregated_score_max(predictions):
    predictions = np.copy(predictions)
    predictions = np.cumsum(predictions, axis=0)
    return score_margin(predictions=predictions)