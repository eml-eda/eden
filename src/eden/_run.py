"""
Currently not working
"""
raise NotImplementedError
from eden._eden_ensemble import _EdenEnsemble
import subprocess


def run(
    *, ensemble: "_EdenEnsemble", target_folder: str = "eden-ensemble", c_args: str = ""
) -> float:
    """
    Compiles, runs and greps the output of a deployed eden ensemble

    Parameters
    ----------
    ensemble : _EdenEnsemble
        Configuration data of the ensemble
    target_folder : str, optional
        Path to the directory with the Makefile, by default "eden-ensemble"
    c_args : str, optional
        Additional string passed to the Makefile, by default ""

    Returns
    -------
    float
        the output of the ensemble

    Raises
    ------
    Exception
        raised in case of issues with compilation or running.
    """
    seconds = 100
    # Compile
    cmd = [
        "cd",
        target_folder,
        "&&",
        "make",
        "clean",
        "all",
        "platform=gvsoc",
        'C_ARGS="',
        c_args,
        '"'
    ]
    output = subprocess.run(
        " ".join(cmd), timeout=seconds, shell=True, capture_output=True, text=True
    )
    if output.returncode != 0:
        raise Exception(f"Compilation failure")
    # Run
    cmd = [
        "cd",
        target_folder,
        "&&",
        "make",
        "run",
        "platform=gvsoc",
        'C_ARGS="',
        c_args,
        '"',
    ]
    output = subprocess.run(
        " ".join(cmd), timeout=seconds, shell=True, capture_output=True, text=True
    )
    # Parse the output
    if output.returncode != 0:
        raise Exception(f"Run failure")

    parsed_output = output.stdout
    parsed_output = parsed_output.split("\n")
    for row in parsed_output:
        if "INFERENCE OUTPUT" in row:
            inference_output = float(row.split(":")[1])

    if "classification" in ensemble.task:
        inference_output = int(inference_output)

    # Report the output
    return inference_output
