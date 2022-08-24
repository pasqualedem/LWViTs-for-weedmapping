import copy
import gc
import os
from typing import Mapping

from wd.experiment.run import run
from wd.experiment.resume import resume_last_run, retrieve_run_to_resume
from wd.utils.grid import make_grid
from wd.utils.utilities import nested_dict_update
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def experiment(settings: Mapping, param_path: str = "local variable"):
    exp_settings = settings['experiment']
    base_grid = settings['parameters']
    other_grids = settings['other_grids']
    track_dir = exp_settings['tracking_dir']
    resume = exp_settings['resume']

    exp_log = open(os.path.join(track_dir if track_dir is not None else '', 'exp_log.txt'), 'a')
    exp_log.write('---\n')
    exp_log.flush()

    if exp_settings['excluded_files']:
        os.environ['WANDB_IGNORE_GLOBS'] = exp_settings['excluded_files']

    logger.info(f'Loaded parameters from {param_path}')

    complete_grids = [base_grid]
    if other_grids:
        complete_grids += \
            [nested_dict_update(copy.deepcopy(base_grid), other_run) for other_run in other_grids]
    logger.info(f'There are {len(complete_grids)} grids')

    grids = [make_grid(grid) for grid in complete_grids]

    if resume:
        starting_grid, starting_run, resume_last = retrieve_run_to_resume(exp_settings, grids)
    else:
        resume_last = False
        starting_grid = exp_settings['start_from_grid']
        starting_run = exp_settings['start_from_run']

    for i, grid in enumerate(grids):
        info = f'Found {len(grid)} runs from grid {i}'
        if i < starting_grid:
            info += f', skipping grid {i} with {len(grid)} runs'
        logger.info(info)

    total_runs = sum(len(grid) for grid in grids)
    total_runs_excl_grid = total_runs - sum([len(grid) for grid in grids[starting_grid:]])
    total_runs_excl = total_runs_excl_grid + starting_run
    total_runs_to_run = total_runs - total_runs_excl
    logger.info(f'Total runs found:              {total_runs}')
    logger.info(f'Total runs excluded by grids:  {total_runs_excl_grid}')
    logger.info(f'Total runs excluded:           {total_runs_excl}')
    logger.info(f'Total runs to run:             {total_runs_to_run}')

    continue_with_errors = exp_settings.pop('continue_with_errors')

    if resume_last:
        logger.info("+ another run to finish!")
        resume_last_run(exp_settings)
    for i in range(starting_grid, len(grids)):
        grid = grids[i]
        if i != starting_grid:
            starting_run = 0
        for j in range(starting_run, len(grid)):
            params = grid[j]
            try:
                exp_log.write(f'{i} {j},')
                logger.info(f'Running grid {i} out of {len(grids) - 1}')
                logger.info(f'Running run {j} out of {len(grid) - 1} '
                            f'({sum([len(grids[k]) for k in range(i)]) + j} / {total_runs - 1})')
                run({'experiment': exp_settings, **params})
                exp_log.write(f' finished \n')
                exp_log.flush()
                gc.collect()
            except Exception as e:
                logger.error(f'Experiment {i} failed with error {e}')
                exp_log.write(f'{i} {j}, crashed \n')
                exp_log.flush()
                if not continue_with_errors:
                    raise e
    exp_log.close()


