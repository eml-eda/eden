import numpy as np
import pandas as pd
import subprocess
import multiprocessing as mp


class Profiler:
    def __init__(self):
        pass

    def run_ensemble(
        self, compile_path: str, lib: str, c_args: str = "", bdir: str = "BUILD"
    ):
        """
        Compile and run an Ensemble and then collect the statistics,
        return a dictionary with the stats and the stderr
        """
        command = [
            "cd",
            compile_path,
            "&&",
            "timeout",
            "15s",  # Avoids hanging inputs
            "make",
            "clean",
            "all",
            "run",
            f"lib={lib}",
            f'c_args="{c_args}"',
            f"BDIR={bdir}",
        ]
        command_output = subprocess.run(
            " ".join(command), shell=True, capture_output=True, text=True
        )  # .stdout.decode("utf-8")
        # Parse the output
        command_output_stderr = command_output.stderr.split("\n")
        command_output_stdout = command_output.stdout.split("\n")
        # Find the statistics
        try:
            idx_start = command_output_stdout.index("Stats inference - start")
            idx_end = command_output_stdout.index("Stats inference - end")
            stats_list = command_output_stdout[idx_start + 1 : idx_end]
            stats_dict = {
                e.split("=")[0].rstrip(): int(e.split("=")[1].rstrip())
                for e in stats_list
            }
        except ValueError:
            stats_dict = {}
            print("Crash detected", c_args)
        return stats_dict, command_output_stderr

    def run_parallel_ensemble(
        self, compile_path, n_inputs, lib, c_args: str = "", n_jobs: int = 1
    ):
        """
        Parallel runs over the inputs
        """
        pools = mp.Pool(n_jobs)
        pools_index = np.arange(n_inputs)
        idx_per_job = np.array_split(np.arange(n_inputs), n_jobs)
        # Size checking
        # Una compilazione singola per vedere se ci sta in memoria
        test_c_args = c_args + f" -DINPUT0"
        stats_dict, command_output_stderr = self.run_ensemble(
            compile_path=compile_path, lib=lib, c_args=test_c_args, bdir="BUILDTEST"
        )
        if stats_dict == {}:
            print("Error in test, halting")
            print(command_output_stderr)
            return pd.DataFrame()
        arguments = [
            (
                compile_path,
                p,
                lib,
                idx_per_job[p],
                c_args,
            )
            for p in range(n_jobs)
        ]
        pools_stats = pools.starmap(self._run_parallel_ensemble, arguments)
        stats = list()

        for result, error in pools_stats:
            for el in error:
                if el != list([""]):
                    print(el)
            stats.extend(result)
        return pd.DataFrame(stats)

    def _run_parallel_ensemble(self, compile_path, pool_id, lib, idx_pool, c_args):
        stats = list()
        errors = list()
        pool_bdir = f"BUILD{pool_id}"
        for idx in idx_pool:
            local_c_args = c_args + f" -DINPUT{idx}"
            run_stats, run_errors = self.run_ensemble(
                compile_path=compile_path, lib=lib, c_args=local_c_args, bdir=pool_bdir
            )
            run_stats["input_idx"] = idx
            stats.append(run_stats)
            errors.append(run_errors)
        return stats, errors
