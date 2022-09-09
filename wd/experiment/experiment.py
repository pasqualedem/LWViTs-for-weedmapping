import copy
import gc
import os
from typing import Mapping
from easydict import EasyDict

from experiment.resume import ExpLog
from wd.experiment.run import Run
from wd.experiment.resume import get_interrupted_run, retrieve_run_to_resume
from wd.utils.grid import make_grid, linearize
from wd.utils.utilities import nested_dict_update, update_collection
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class GridSummary:
    def __init__(self,
                 total_runs,
                 total_runs_excl_grid,
                 total_runs_to_run,
                 total_runs_excl,
                 ):
        self.total_runs = total_runs
        self.total_runs_excl_grid = total_runs_excl_grid
        self.total_runs_to_run = total_runs_to_run
        self.total_runs_excl = total_runs_excl

    def update(self, d):
        self.total_runs = d.get("total_runs") or self.total_runs
        self.total_runs_excl_grid = d.get("total_runs_excl_grid") or self.total_runs_excl_grid
        self.total_runs_to_run = d.get("total_runs_to_run") or self.total_runs_to_run
        self.total_runs_excl = d.get("total_runs_excl") or self.total_runs_to_run


class ExpSettings(EasyDict):
    def __init__(self, *args, **kwargs):
        self.start_from_grid = 0
        self.start_from_run = 0
        self.resume = False
        self.resume_last = False
        self.tracking_dir = ""
        self.excluded_files = ""
        super().__init__(*args, **kwargs)


class Experimenter:
    def __init__(self):
        self.gs = None
        self.exp_settings = ExpSettings()
        self.grids = None

    def calculate_runs(self, settings):
        base_grid = settings['parameters']
        other_grids = settings['other_grids']
        self.exp_settings = ExpSettings(settings['experiment'])

        complete_grids = [base_grid]
        if other_grids:
            complete_grids += \
                [nested_dict_update(copy.deepcopy(base_grid), other_run) for other_run in other_grids]
        logger.info(f'There are {len(complete_grids)} grids')

        self.grids, dot_elements = zip(*[make_grid(grid, return_cartesian_elements=True) for grid in complete_grids])
        dot_elements = list(dot_elements)
        dot_elements[1:] = [list(dict(linearize(others) + dot).items()) for others, dot in
                            zip(other_grids, dot_elements[1:])]

        # Modify starting grid and run to manage the resume
        self.manage_resume()

        for i, grid in enumerate(self.grids):
            info = f'Found {len(grid)} runs from grid {i}'
            if i < self.exp_settings.start_from_grid:
                info += f', skipping grid {i} with {len(grid)} runs'
            logger.info(info)
        self.generate_grid_summary()
        return self.gs, self.grids, dot_elements

    def generate_grid_summary(self):
        total_runs = sum(len(grid) for grid in self.grids)
        total_runs_excl_grid = total_runs - sum([len(grid) for grid in self.grids[self.exp_settings.start_from_grid:]])
        total_runs_excl = total_runs_excl_grid + self.exp_settings.start_from_run
        total_runs_to_run = total_runs - total_runs_excl
        self.gs = GridSummary(
            total_runs=total_runs,
            total_runs_excl_grid=total_runs_excl_grid,
            total_runs_to_run=total_runs_to_run,
            total_runs_excl=total_runs_excl
        )

    def execute_runs(self, callback=None):
        track_dir = self.exp_settings['tracking_dir']
        exp_log = ExpLog(track_dir)
        exp_log.start()
        starting_run = self.exp_settings.start_from_grid
        if self.exp_settings.resume_last:
            logger.info("+ another run to finish!")
            grid_len = len(self.grids[self.exp_settings.start_from_grid])
            sg = self.exp_settings.start_from_grid
            sr = self.exp_settings.start_from_run - 1
            try:
                exp_log.insert_run(sg, sr)
                run = get_interrupted_run(self.exp_settings)
                if callback:
                    yield callback(self.exp_settings.start_from_grid, self.exp_settings.start_from_run - 1,
                                   len(self.grids), grid_len,
                                   status="started", run_params=run.params)
                logger.info(f'Running grid {sg} out of {len(self.grids) - 1}')
                logger.info(f'Running run {sr - 1} out of {grid_len} '
                            f'({sum([len(self.grids[k]) for k in range(sg)]) + sr} / {self.gs.total_runs - 1})')
                run.launch()
                exp_log.finish_run()
            except Exception as e:
                logger.error(f'Experiment {sg} failed with error {e}')
                exp_log.crash_run()
                if not self.exp_settings.continue_with_errors:
                    raise e
                if callback:
                    yield callback(sg, sr, len(self.grids), grid_len, status="crashed", run_params={}, exception=e)
        for i in range(self.exp_settings.start_from_grid, len(self.grids)):
            grid = self.grids[i]
            if i != self.exp_settings.start_from_grid:
                starting_run = 0
            for j in range(starting_run, len(grid)):
                params = grid[j]
                try:
                    exp_log.insert_run(i, j)
                    if callback:
                        yield callback(i, j, len(self.grids), len(grid), status="started", run_params=params)
                    logger.info(f'Running grid {i} out of {len(self.grids) - 1}')
                    logger.info(f'Running run {j} out of {len(grid) - 1} '
                                f'({sum([len(self.grids[k]) for k in range(i)]) + j} / {self.gs.total_runs - 1})')
                    run = Run()
                    run.init({'experiment': {**self.exp_settings}, **params})
                    run.launch()
                    exp_log.finish_run()
                    gc.collect()
                    if callback:
                        yield callback(i, j, len(self.grids), len(grid), status="finished", run_params={})
                except Exception as e:
                    logger.error(f'Experiment {i} failed with error {e}')
                    exp_log.crash_run()
                    if not self.exp_settings.continue_with_errors:
                        raise e
                    if callback:
                        yield callback(i, j, len(self.grids), len(grid), status="crashed", run_params={}, exception=e)
        exp_log.close()

    def manage_resume(self):
        if self.exp_settings.resume:
            self.exp_settings.start_from_grid, \
                self.exp_settings.start_from_run, \
                self.exp_settings.resume_last = retrieve_run_to_resume(self.exp_settings, self.grids)
        else:
            self.exp_settings.resume_last = False

    def update_settings(self, d):
        self.exp_settings = update_collection(self.exp_settings, d)
        if self.gs is None:
            return
        self.gs.update(self.exp_settings)
        if "resume" in d:
            self.manage_resume()
            self.generate_grid_summary()


def experiment(settings: Mapping, param_path: str = "local variable"):
    exp = Experimenter()
    exp.calculate_runs(settings)
    exp_settings = exp.exp_settings

    if exp_settings.excluded_files:
        os.environ['WANDB_IGNORE_GLOBS'] = exp_settings.excluded_files

    logger.info(f'Loaded parameters from {param_path}')

    experimenter = Experimenter()
    grid_summary, grids, cartesian_elements = experimenter.calculate_runs(settings)

    logger.info(f'Total runs found:              {grid_summary.total_runs}')
    logger.info(f'Total runs excluded by grids:  {grid_summary.total_runs_excl_grid}')
    logger.info(f'Total runs excluded:           {grid_summary.total_runs_excl}')
    logger.info(f'Total runs to run:             {grid_summary.total_runs_to_run}')
    experimenter.execute_runs()
