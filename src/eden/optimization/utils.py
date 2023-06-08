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

from typing import Tuple
import math


def _get_bits_to_represent(
    *,
    range_val: Tuple[int, int],
    return_cvalid: bool = True,
) -> int:
    range_val = range_val[1] - range_val[0]
    n_bits = math.ceil(math.log2(range_val))
    if return_cvalid:
        # Set to multiple of 8
        n_bits = ((n_bits + 7) // 8) * 8
        if n_bits == 24:
            n_bits = 32
        elif n_bits > 32:
            raise NotImplementedError("Ensemble is too large")
    return n_bits
