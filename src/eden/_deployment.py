import os
from eden._eden_ensemble import _EdenEnsemble
from mako.lookup import TemplateLookup
from eden import _formatter
import pkg_resources

TEMPLATES_DIR = pkg_resources.resource_filename("eden", "templates")


def _write_template(
    lookup: TemplateLookup,
    template_name: str,
    template_data: _EdenEnsemble,
    output_folder: str,
):
    output_fname = os.path.join(output_folder, template_name)
    template = lookup.get_template(template_name)
    t = template.render(config=template_data, formatter=_formatter)

    with open(f"{output_fname}", "w") as out_file:
        out_file.write(t)
    file_extension = os.path.splitext(output_fname)[1]
    if file_extension in [".c", ".h"]:
        os.system(f"clang-format -i {output_fname}")


def _deploy(*, output_folder: str, ensemble: _EdenEnsemble) -> None:
    os.makedirs(name=output_folder, exist_ok=True)
    # Common folder first
    lookup = TemplateLookup(
        directories=[
            os.path.join(TEMPLATES_DIR, "common"),
            os.path.join(TEMPLATES_DIR, ensemble.target_architecture),
        ]
    )
    _write_template(
        lookup=lookup,
        template_name="ensemble.h",
        output_folder=output_folder,
        template_data=ensemble,
    )
    _write_template(
        lookup=lookup,
        template_name="ensemble_data.h",
        output_folder=output_folder,
        template_data=ensemble,
    )
    _write_template(
        lookup=lookup,
        template_name="input.h",
        output_folder=output_folder,
        template_data=ensemble,
    )
    for e in os.listdir(os.path.join(TEMPLATES_DIR, ensemble.target_architecture)):
        full_path = os.path.join(TEMPLATES_DIR, ensemble.target_architecture, e)
        if os.path.isfile(full_path):
            _write_template(
                lookup=lookup,
                template_name=e,
                output_folder=output_folder,
                template_data=ensemble,
            )
