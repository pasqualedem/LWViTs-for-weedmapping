import pandas as pd

from wd.utils.grid import linearized_to_string
from wd.experiment.experiment import GridSummary, Experimenter


def exp_summary_builder(exp: Experimenter):
    txt = ""
    txt += "|Property | Value |\n"
    txt += "|---------|-------|\n"
    txt += f"|Resume experiment            | {exp.exp_settings.resume} |\n"
    txt += f"|Resume interrupted run       | {exp.exp_settings.resume_last} |\n"
    txt += f"|Continue with errors         | {exp.exp_settings.continue_with_errors} |\n"
    txt += f"|Excluded files               | {exp.exp_settings.excluded_files} |\n"
    txt += f"|Starting grid                | {exp.exp_settings.start_from_grid} |\n"
    txt += f"|Starting run                 | {exp.exp_settings.start_from_run} |\n"
    txt += f"|Total runs                   | {exp.gs.total_runs} |\n"
    txt += f"|Total runs to run            | {exp.gs.total_runs_to_run} |\n"
    txt += f"|Total runs excluded by grids | {exp.gs.total_runs_excl_grid} |\n"
    txt += f"|Total runs excluded          | {exp.gs.total_runs_excl} |\n"

    return txt


def grid_summary_builder(grids, dot_elements):
    dfs = [pd.DataFrame(linearized_to_string(dot_element), columns=[f"Grid {i}", f"N. runs: {len(grid)}"])
           for i, (dot_element, grid) in enumerate(zip(dot_elements, grids))]
    mark_grids = "\n\n".join(df.to_markdown(index=False) for df in dfs)

    return mark_grids


def title_builder(title):
    return f"## {title}"


class MkFailures:
    def __init__(self):
        self.value = "## Failures \n\n"

    def get_text(self):
        return self.value

    def update(self, cur_grid, cur_run, exception):
        self.value += f"Grid {cur_grid} Run {cur_run} failed with an exception: <br>"
        self.value += f"{exception} \n\n ---"
