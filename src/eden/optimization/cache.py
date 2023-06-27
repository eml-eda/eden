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

"""
Collections of functions to determine the L-Flag for each variable,
it requires an estimation of the memory
"""
import math
from typing import Mapping, Tuple
from copy import deepcopy
import numpy as np


def _knapsack_01(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    selected = [[False] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] > w:
                dp[i][w] = dp[i - 1][w]
            else:
                included_value = values[i - 1] + dp[i - 1][w - weights[i - 1]]
                excluded_value = dp[i - 1][w]

                if included_value > excluded_value:
                    dp[i][w] = included_value
                    selected[i][w] = True
                else:
                    dp[i][w] = excluded_value

    selected_items = []
    w = capacity
    for i in range(n, 0, -1):
        if selected[i][w]:
            selected_items.append(i - 1)
            w -= weights[i - 1]

    selected_items.reverse()
    return dp[n][capacity], selected_items


# For now it is used only by GAP8
# Set in L1 everything fitting, leave the rest in L2 (no flag for GAP8)
def _cache_placer(**kwargs: Mapping[str, Tuple[int, int]]) -> Mapping:
    # TODO : Document this
    # The order depends on the access ratio.
    SIZE_L1 = 64000
    buffers_to_place = deepcopy(kwargs)
    weights = list()
    access_rates = list()

    # Knapsack
    for buffer_name, data in buffers_to_place.items():
        weights.append(data[0])
        access_rates.append(data[1])

    _, idx_in_sack = _knapsack_01(
        weights=weights, values=access_rates, capacity=SIZE_L1
    )

    buffer_placement = list()
    for idx in range(len(buffers_to_place.keys())):
        if idx in idx_in_sack:
            buffer_placement.append("L1")
        else:
            buffer_placement.append("")

    return buffer_placement
