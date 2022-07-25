import argparse
import json
import os
from typing import Mapping

import torch
import gc
import copy

import numpy as np
import wandb

from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from ruamel.yaml import YAML

from wd.utils.utilities import nested_dict_update, dict_to_yaml_string, update_collection
from wd.callbacks import SegmentationVisualizationCallback, WandbCallback
from wd.data.sequoia import WeedMapDatasetInterface
from wd.loss import LOSSES as LOSSES_DICT
from wd.metrics import metrics_factory
from wd.utils.grid import make_grid
from wd.utils.utilities import values_to_number
from wd.learning.seg_trainer import SegmentationTrainer
from wd.learning.wandb_logger import WandBSGLogger

logger = get_logger(__name__)


def parse_params(params: dict) -> (dict, dict, dict, list):
    # Set Random seeds
    torch.manual_seed(params['train_params']['seed'])
    np.random.seed(params['train_params']['seed'])

    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params']['loss']
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    train_metrics = metrics_factory(params['train_metrics'])
    test_metrics = metrics_factory(params['test_metrics'])

    # dataset params
    dataset_params = params['dataset']

    train_params = {
        **input_train_params,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss": loss,
        "loss_logging_items_names": ["loss"],
        "sg_logger": WandBSGLogger,
        'sg_logger_params': {
            'entity': params['experiment']['entity'],
            'tags': params['tags'],
            'project_name': params['experiment']['name'],
        }
    }

    test_params = {
        "test_metrics": test_metrics,
    }

    # early stopping
    early_stop = [EarlyStop(Phase.VALIDATION_EPOCH_END, **params['early_stopping']['params'])] \
        if params['early_stopping']['enabled'] else []

    return train_params, test_params, dataset_params, early_stop


def run(params: dict):
    seg_trainer = None
    try:
        phases = params['phases']

        train_params, test_params, dataset_params, early_stop = parse_params(params)

        seg_trainer = SegmentationTrainer(experiment_name=params['experiment']['group'],
                                          ckpt_root_dir=params['experiment']['tracking_dir']
                                          if params['experiment']['tracking_dir'] else 'wandb')
        dataset = WeedMapDatasetInterface(dataset_params)
        seg_trainer.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])
        seg_trainer.init_model(params, False, None)
        seg_trainer.init_loggers({"in_params": params}, train_params)
        logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")

        if 'train' in phases:
            train(seg_trainer, train_params, dataset, early_stop)

        if 'test' in phases:
            test_metrics = seg_trainer.test(**test_params)

        if 'inference' in phases:
            inference(seg_trainer, params['run_params'], dataset)
    finally:
        if seg_trainer is not None:
            seg_trainer.sg_logger.close(True)


def train(seg_trainer, train_params, dataset, early_stop):
    # ------------------------ TRAINING PHASE ------------------------
    # Callbacks
    cbcks = [
        WandbCallback(Phase.TRAIN_EPOCH_END, freq=1),
        WandbCallback(Phase.VALIDATION_EPOCH_END, freq=1),
        SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                          freq=1,
                                          batch_idxs=[0, len(seg_trainer.train_loader) - 1],
                                          last_img_idx_in_batch=4,
                                          num_classes=dataset.trainset.CLASS_LABELS,
                                          undo_preprocessing=dataset.undo_preprocess),
        *early_stop
    ]
    train_params["phase_callbacks"] = cbcks

    seg_trainer.train(train_params)
    gc.collect()


def inference(seg_trainer, run_params, dataset):
    run_loader = dataset.get_run_loader(folders=run_params['run_folders'], batch_size=run_params['batch_size'])
    cbcks = [
        # SaveSegmentationPredictionsCallback(phase=Phase.POST_TRAINING,
        #                                     path=
        #                                     run_params['prediction_folder']
        #                                     if run_params['prediction_folder'] != 'mlflow'
        #                                     else mlclient.run.info.artifact_uri + '/predictions',
        #                                     num_classes=len(seg_trainer.test_loader.dataset.classes),
        #                                     )
    ]
    run_loader.dataset.return_name = True
    seg_trainer.run(run_loader, callbacks=cbcks)
    # seg_trainer.valid_loader.dataset.return_name = True
    # seg_trainer.run(seg_trainer.valid_loader, callbacks=cbcks)


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

    if resume:
        starting_grid, starting_run, resume_last = retrieve_run_to_resume(exp_settings, complete_grids)
        if resume_last:
            resume_last_run(exp_settings)
    else:
        starting_grid = exp_settings['start_from_grid']
        starting_run = exp_settings['start_from_run']

    grids = []
    for i, grid in enumerate(complete_grids):
        grid_runs = make_grid(grid)
        info = f'Found {len(grid_runs)} runs from grid {i}'
        if i < starting_grid:
            info += f', skipping grid {i} with {len(grid_runs)} runs'
        grids = grids + [grid_runs]
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


def retrieve_run_to_resume(settings, grids):
    grid_list = [(i, j) for i in range(len(grids)) for j in range(len(grids[i]))]
    dir = settings['tracking_dir']
    dir_file_path = os.path.join(dir if dir is not None else '', 'exp_log.txt')
    with open(dir_file_path, 'r') as f:
        last_ran = f.readlines()[-1]
    code, status = last_ran.split(",")
    i, j = map(int, code.split(" "))
    index = grid_list.index((i, j))
    try:
        start_grid, start_run = grid_list[index + 1]
    except IndexError as e:
        if status == "finished \n":
            logger.info(e)
            raise ValueError('No experiment to resume!!')
        else:
            return len(grids), None, True
    resume_last = True if status != "finished \n" else False
    return start_grid, start_run, resume_last


def resume_last_run(input_settings):
    namespace = input_settings["name"]
    group = input_settings["group"]
    last_run = wandb.Api().runs(path=namespace, filters={"group": group,}, order="-created_at")
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


parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('--resume_run', required=False, action='store_true',
                    help='Resume the run(s)', default=False)
parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the experiment', default=False)

parser.add_argument('-d', '--dir', required=False, type=str,
                    help='Set the local tracking directory', default=None)

parser.add_argument("--grid", type=int, help="Select the first grid to start from")
parser.add_argument("--run", type=int, help="Select the run in grid to start from")

parser.add_argument('-f', "--filters", type=json.loads, help="Filters to query in the resuming mode")
parser.add_argument('-s', "--stage", type=json.loads, help="Stages to execute in the resuming mode")
parser.add_argument('-p', "--path", type=str, help="Path to the tracking url in the resuming mode")

if __name__ == '__main__':
    args = parser.parse_args()

    track_dir = args.dir
    filters = args.filters
    stage = args.stage
    path = args.path

    if args.resume_run:
        param_path = 'resume.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)
        settings['runs'][0]['filters'] = update_collection(settings['runs'][0]['filters'], filters)
        settings['runs'][0]['stage'] = update_collection(settings['runs'][0]['stage'], stage)
        settings = update_collection(settings, path, key="path")
        resume_run(settings)
    else:
        param_path = 'parameters.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)
        settings['experiment'] = update_collection(settings['experiment'], args.resume, key='resume')
        settings['experiment'] = update_collection(settings['experiment'], args.grid, key='start_from_grid')
        settings['experiment'] = update_collection(settings['experiment'], args.run, key='start_from_run')
        settings['experiment'] = update_collection(settings['experiment'], track_dir, key='tracking_dir')
        experiment(settings)
