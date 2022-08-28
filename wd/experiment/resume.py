import os

import wandb

from wd.data.sequoia import WeedMapDatasetInterface
from wd.experiment.run import train
from wd.experiment.parameters import parse_params
from wd.learning.seg_trainer import SegmentationTrainer
from wd.utils.utilities import values_to_number, nested_dict_update

from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


def resume_run(settings):
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        filters = query['filters']
        stage = query['stage']
        updated_config = query['updated_config']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        if len(runs) == 0:
            logger.error(f'No runs found for query {filters}')
        for run in runs:
            seg_trainer = None
            try:
                params = values_to_number(run.config['in_params'])
                params = nested_dict_update(params, updated_config)
                run.config['in_params'] = params
                run.update()
                train_params, test_params, dataset_params, early_stop = parse_params(params)

                seg_trainer = SegmentationTrainer(experiment_name=params['experiment']['group'],
                                                  ckpt_root_dir=params['experiment']['tracking_dir']
                                                  if params['experiment']['tracking_dir'] else 'wandb')
                dataset = WeedMapDatasetInterface(dataset_params)
                seg_trainer.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])
                track_dir = run.config.get('in_params').get('experiment').get('tracking_dir') or 'wandb'
                checkpoint_path_group = os.path.join(track_dir, run.group, 'wandb')
                run_folder = list(filter(lambda x: str(run.id) in x, os.listdir(checkpoint_path_group)))
                ckpt = 'ckpt_latest.pth' if 'train' in stage else 'ckpt_best.pth'
                checkpoint_path = os.path.join(checkpoint_path_group, run_folder[0], 'files', ckpt)
                seg_trainer.init_model(params, True, checkpoint_path)
                seg_trainer.init_loggers({"in_params": params}, train_params, run_id=run.id)
                if 'train' in stage:
                    train(seg_trainer, train_params, dataset, early_stop)
                elif 'test' in stage:
                    test_metrics = seg_trainer.test(**test_params)
            finally:
                if seg_trainer is not None:
                    seg_trainer.sg_logger.close(really=True)


def resume_last_run(input_settings):
    namespace = input_settings["name"]
    group = input_settings["group"]
    last_run = wandb.Api().runs(path=namespace, filters={"group": group,}, order="-created_at")[0]
    resume_settings = {
        "path": namespace,
        "runs": [
            {
                "filters": {"group": group, "name": last_run.id},
                "stage": ["train", "test"],
                "updated_config": None,
                "updated_meta": None
            }
        ]
    }
    resume_run(resume_settings)


def retrieve_run_to_resume(settings, grids):
    grid_list = [(i, j) for i in range(len(grids)) for j in range(len(grids[i]))]
    dir = settings['tracking_dir']
    dir_file_path = os.path.join(dir if dir is not None else '', 'exp_log.txt')
    with open(dir_file_path, 'r') as f:
        lines = f.readlines()
        i = 1
        while lines[-i] == '---\n':
            i += 1
        last_ran = lines[-i]

    code, status = last_ran.split(",")
    i, j = map(int, code.split(" "))
    index = grid_list.index((i, j))
    try:
        start_grid, start_run = grid_list[index + 2]  # Skip interrupted run
    except IndexError as e:
        if status == "finished \n":
            logger.info(e)
            raise ValueError('No experiment to resume!!')
        else:
            return len(grids), None, True
    resume_last = True if status != "finished \n" else False
    return start_grid, start_run, resume_last
