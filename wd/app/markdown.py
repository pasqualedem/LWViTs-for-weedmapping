import pandas as pd

from utils.grid import linearized_to_string
from wd.experiment.experiment import GridSummary


def grid_summary_builder(grid_summary: GridSummary, grids, dot_elements):
    txt = ""
    txt += "|Property | Value |\n"
    txt += "|---------|-------|\n"
    txt += f"|Starting grid                | {grid_summary.starting_grid} |\n"
    txt += f"|Starting run                 | {grid_summary.starting_run} |\n"
    txt += f"|Total runs                   | {grid_summary.total_runs} |\n"
    txt += f"|Total runs to run            | {grid_summary.total_runs_to_run} |\n"
    txt += f"|Total runs excluded by grids | {grid_summary.total_runs_excl_grid} |\n"
    txt += f"|Total runs excluded          | {grid_summary.total_runs_excl} |\n"
    txt += f"|Resume last                  | {grid_summary.resume_last} |\n"

    dfs = [pd.DataFrame(linearized_to_string(dot_element), columns=[f"Grid {i}", f"N. runs: {len(grid)}"])
           for i, (dot_element, grid) in enumerate(zip(dot_elements, grids))]
    mark_grids = "\n\n".join(df.to_markdown(index=False) for df in dfs)

    return txt, mark_grids
