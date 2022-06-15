import argparse
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

from utils.utils import nested_dict_update, dict_to_yaml_string
from wd.callbacks import SegmentationVisualizationCallback, WandbCallback
from wd.data.sequoia import WeedMapDatasetInterface
from wd.loss import LOSSES as LOSSES_DICT
from wd.metrics import metrics_factory
from wd.utils.grid import make_grid
from wd.utils.utils import values_to_number
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
    starting_grid = exp_settings['start_from_grid']
    starting_run = exp_settings['start_from_run']

    exp_log = open('exp_log.txt', 'a')
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
                logger.info(f'Running grid {i} out of {len(grids) - 1}')
                logger.info(f'Running run {j} out of {len(grid) - 1} '
                            f'({sum([len(grids[k]) for k in range(i)]) + j} / {total_runs - 1})')
                run({'experiment': exp_settings, **params})
                exp_log.write(f'{i} {j}, finished \n')
                exp_log.flush()
                gc.collect()
            except Exception as e:
                logger.error(f'Experiment {i} failed with error {e}')
                exp_log.write(f'{i} {j}, crashed \n')
                exp_log.flush()
                if not continue_with_errors:
                    raise e
    exp_log.close()


def resume(settings):
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
                checkpoint_path_group = os.path.join('wandb', run.group, 'wandb')
                run_folder = list(filter(lambda x: str(run.id) in x, os.listdir(checkpoint_path_group)))
                checkpoint_path = os.path.join(checkpoint_path_group, run_folder[0], 'files', 'ckpt_best.pth')
                seg_trainer.init_model(params, True, checkpoint_path)
                seg_trainer.init_loggers({"in_params": params}, train_params, run_id=run.id)
                if stage == 'train':
                    train(seg_trainer, train_params, dataset, early_stop)
                elif stage == 'test':
                    test_metrics = seg_trainer.test(**test_params)
            finally:
                if seg_trainer is not None:
                    seg_trainer.sg_logger.close(really=True)


parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the run(s)', default=False)

if __name__ == '__main__':
    args = parser.parse_args()

    if args.resume:
        param_path = 'resume.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)
        resume(settings)
    else:
        param_path = 'parameters.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)

        experiment(settings)
