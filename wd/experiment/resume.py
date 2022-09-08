import os

import wandb

from wd.data.sequoia import WeedMapDatasetInterface
from wd.experiment.run import train, Run
from wd.experiment.parameters import parse_params
from wd.learning.seg_trainer import SegmentationTrainer
from wd.utils.utilities import values_to_number, nested_dict_update

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def resume_set_of_runs(settings, post_filters=None):
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        filters = query['filters']
        stage = query['stage']
        updated_config = query['updated_config']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        runs = list(filter(post_filters, runs))
        print("Runs to resume:")
        for run in runs:
            print(f"{run.group} \t - \t {run.name}")
        if len(runs) == 0:
            logger.error(f'No runs found for query {filters} with post_filters: {post_filters}')
        for run in runs:
            resume_run(run, updated_config, stage)


def complete_incompleted_runs(settings):
    print("Going on to complete runs!")
    resume_set_of_runs(settings, lambda x: 'f1' not in x.summary)


def resume_run(wandb_run, updated_config, stage):
    to_resume_run = Run()
    to_resume_run.resume(wandb_run=wandb_run, updated_config=updated_config, phases=stage)
    to_resume_run.launch()


def get_interrupted_run(input_settings):
    namespace = input_settings["name"]
    group = input_settings["group"]
    last_run = wandb.Api().runs(path=namespace, filters={"group": group,}, order="-created_at")[0]
    filters = {"group": group, "name": last_run.id}
    stage = ["train", "test"]
    updated_config = None
    api = wandb.Api()
    runs = api.runs(path=namespace, filters=filters)
    if len(runs) == 0:
        raise RuntimeError("No runs found")
    if len(runs) > 1:
        raise EnvironmentError("More than 1 run???")
    to_resume_run = Run()
    to_resume_run.resume(wandb_run=runs[0], updated_config=updated_config, phases=stage)
    return to_resume_run


def retrieve_run_to_resume(settings, grids):
    grid_list = [(i, j) for i in range(len(grids)) for j in range(len(grids[i]))]
    dir = settings['tracking_dir']
    exp_log = ExpLog(track_dir=dir, mode="r")
    i, j, status = exp_log.get_last_run()
    index = grid_list.index((i, j))
    try:
        start_grid, start_run = grid_list[index + 1]  # Skip interrupted run
    except IndexError as e:
        if status == "finished \n":
            logger.info(e)
            raise ValueError('No experiment to resume!!')
        else:
            return len(grids), None, True
    resume_last = True if status != "finished \n" else False
    return start_grid, start_run, resume_last


class ExpLog:
    EXP_END = '---\n'
    FINISHED = 'finished \n'
    CRASHED = 'crashed \n'

    def __init__(self, track_dir, mode='a'):
        self.exp_log = open(os.path.join(track_dir if track_dir is not None else '', 'exp_log.txt'), mode)

    def start(self):
        self.exp_log.write(self.EXP_END)
        self.exp_log.flush()

    def write(self, s):
        self.exp_log.write(s)
        self.exp_log.flush()

    def close(self):
        self.exp_log.close()

    def insert_run(self, i, j):
        self.exp_log.write(f'{i} {j},')
        self.exp_log.flush()

    def finish_run(self):
        self.exp_log.write(self.FINISHED)
        self.exp_log.flush()

    def crash_run(self):
        self.exp_log.write(self.CRASHED)
        self.exp_log.flush()

    def get_last_run(self):
        lines = self.exp_log.readlines()
        i = 1
        while lines[-i] in [self.EXP_END, '\n']:
            i += 1
        last_ran = lines[-i]

        code, status = last_ran.split(",")
        i, j = map(int, code.split(" "))
        return i, j, status