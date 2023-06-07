from typing import Mapping, List
import numpy as np
import os
from ._formatter import to_c_array2d, to_c_array, to_node_struct


def _write_template_input(
    lookup, output_dir, template_data_src: Mapping, test_data: List[np.ndarray]
):
    os.makedirs(os.path.join(output_dir, "autogen"), exist_ok=True)
    output_fname = os.path.join(output_dir, "autogen", "eden_input.h")
    template = lookup.get_template("eden_input.h")

    t = template.render(input_list=[to_c_array(t) for t in test_data])

    with open(f"{output_fname}", "w") as out_file:
        out_file.write(t)
    file_extension = os.path.splitext(output_fname)[1]
    if file_extension in [".c", ".h"]:
        os.system(f"clang-format -i {output_fname}")
    return template_data_src
