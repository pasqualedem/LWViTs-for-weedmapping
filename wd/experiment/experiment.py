import copy
import gc
import os
from typing import Mapping

from wd.experiment.run import run
from wd.experiment.resume import resume_last_run, retrieve_run_to_resume
from wd.utils.grid import make_grid, linearize
from wd.utils.utilities import nested_dict_update
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class GridSummary:
    def __init__(self,
                 total_runs,
                 total_runs_excl_grid,
                 total_runs_to_run,
                 total_runs_excl,
                 resume_last,
                 starting_grid,
                 starting_run,
                 ):
        self.total_runs = total_runs
        self.total_runs_excl_grid = total_runs_excl_grid
        self.total_runs_to_run = total_runs_to_run
        self.total_runs_excl = total_runs_excl
        self.resume_last = resume_last
        self.starting_grid = starting_grid
        self.starting_run = starting_run


class Experimenter:
    def __init__(self):
        self.gs = None
        self.exp_settings = None
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings['parameters']
        other_grids = settings['other_grids']
        self.exp_settings = settings['experiment']
        resume = self.exp_settings.get('resume') or False

        complete_grids = [base_grid]
        if other_grids:
            complete_grids += \
                [nested_dict_update(copy.deepcopy(base_grid), other_run) for other_run in other_grids]
        logger.info(f'There are {len(complete_grids)} grids')

        grids, dot_elements = zip(*[make_grid(grid, return_cartesian_elements=True) for grid in complete_grids])
        dot_elements = list(dot_elements)
        dot_elements[1:] = [list(dict(linearize(others) + dot).items()) for others, dot in zip(other_grids, dot_elements[1:])]

        if resume:
            starting_grid, starting_run, resume_last = retrieve_run_to_resume(self.exp_settings, grids)
        else:
            resume_last = False
            starting_grid = self.exp_settings['start_from_grid']
            starting_run = self.exp_settings['start_from_run']

        for i, grid in enumerate(grids):
            info = f'Found {len(grid)} runs from grid {i}'
            if i < starting_grid:
                info += f', skipping grid {i} with {len(grid)} runs'
            logger.info(info)

        total_runs = sum(len(grid) for grid in grids)
        total_runs_excl_grid = total_runs - sum([len(grid) for grid in grids[starting_grid:]])
        total_runs_excl = total_runs_excl_grid + starting_run
        total_runs_to_run = total_runs - total_runs_excl
        self.grids = grids
        self.gs = GridSummary(
            total_runs=total_runs,
            total_runs_excl_grid=total_runs_excl_grid,
            total_runs_to_run=total_runs_to_run,
            total_runs_excl=total_runs_excl,
            resume_last=resume_last,
            starting_grid=starting_grid,
            starting_run=starting_run
        )
        return self.gs, grids, dot_elements

    def execute_runs(self, callback=None):
        continue_with_errors = self.exp_settings.pop('continue_with_errors')
        track_dir = self.exp_settings['tracking_dir']
        exp_log = ExpLog(track_dir)

        starting_run = self.gs.starting_run
        if self.gs.resume_last:
            logger.info("+ another run to finish!")
            resume_last_run(self.exp_settings)
        for i in range(self.gs.starting_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.gs.starting_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    exp_log.write(f'{i} {j},')
                    logger.info(f'Running grid {i} out of {len(self.grids) - 1}')
                    logger.info(f'Running run {j} out of {len(grid) - 1} '
                                f'({sum([len(self.grids[k]) for k in range(i)]) + j} / {self.gs.total_runs - 1})')
                    run({'experiment': self.exp_settings, **params})
                    exp_log.write(f' finished \n')
                    gc.collect()
                    if callback:
                        yield callback(i, j, len(self.grids), len(grid), success=True)
                except Exception as e:
                    logger.error(f'Experiment {i} failed with error {e}')
                    exp_log.write(f'{i} {j}, crashed \n')
                    if not continue_with_errors:
                        raise e
                    if callback:
                        yield callback(i, j, len(self.grids), len(grid), success=False)
        exp_log.close()


def experiment(settings: Mapping, param_path: str = "local variable"):
    exp_settings = settings['experiment']

    if exp_settings['excluded_files']:
        os.environ['WANDB_IGNORE_GLOBS'] = exp_settings['excluded_files']

    logger.info(f'Loaded parameters from {param_path}')

    experimenter = Experimenter()
    grid_summary, grids, cartesian_elements = experimenter.calculate_runs(settings)

    logger.info(f'Total runs found:              {grid_summary.total_runs}')
    logger.info(f'Total runs excluded by grids:  {grid_summary.total_runs_excl_grid}')
    logger.info(f'Total runs excluded:           {grid_summary.total_runs_excl}')
    logger.info(f'Total runs to run:             {grid_summary.total_runs_to_run}')
    experimenter.execute_runs(grid_summary, grids)


class ExpLog:
    def __init__(self, track_dir):
        self.exp_log = open(os.path.join(track_dir if track_dir is not None else '', 'exp_log.txt'), 'a')
        self.exp_log.write('---\n')
        self.exp_log.flush()

    def write(self, s):
        self.exp_log.write(s)
        self.exp_log.flush()

    def close(self):
        self.exp_log.close()
