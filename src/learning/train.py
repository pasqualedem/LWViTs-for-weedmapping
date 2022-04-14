import os
from typing import Any, Mapping

import torch
import gc

import numpy as np


from learning.seg_trainer import SegmentationTrainer
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from ruamel.yaml import YAML

from callbacks import SegmentationVisualizationCallback, MlflowCallback, SaveSegmentationPredictionsCallback
from data.sequoia import SequoiaDatasetInterface

from loss import LOSSES as LOSSES_DICT
from metrics import metrics_factory
from utils.utils import MLRun
from utils.grid import make_grid

torch.manual_seed(42)
np.random.seed(42)
EXP_DIR = 'mlflow'
logger = get_logger(__name__)


def parse_params(params: dict) -> (dict, dict, dict, list):
    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params'].pop('loss')
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    train_metrics = metrics_factory(params['train_metrics'])
    test_metrics = metrics_factory(params['test_metrics'])

    # dataset params
    dataset_params = params['dataset']

    train_params = {
        "greater_metric_to_watch_is_better": True,
        "train_metrics_list": list(train_metrics.values()),
        "valid_metrics_list": list(test_metrics.values()),
        "loss_logging_items_names": ["loss"],
        "loss": loss,
        **input_train_params
    }

    test_params = {
        "test_metrics": test_metrics,
    }

    # early stopping
    early_stop = [EarlyStop(Phase.VALIDATION_EPOCH_END, **params['early_stopping']['params'])] \
        if params['early_stopping']['enabled'] else []

    return train_params, test_params, dataset_params, early_stop


def run(params: dict):
    exp_name = params['name']
    description = params['description']
    phase = params['phase']

    train_params, test_params, dataset_params, early_stop = parse_params(params)

    # Mlflow
    if not (phase == 'train'):
        run_hash = params['run_hash']
    else:
        run_hash = None
    mlclient = MLRun(exp_name, description, run_hash)

    seg_trainer = SegmentationTrainer(experiment_name='SG', ckpt_root_dir=mlclient.run.info.artifact_uri)
    dataset = SequoiaDatasetInterface(dataset_params)
    seg_trainer.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])
    seg_trainer.init_model(params, phase, mlclient)

    if phase == 'train':
        # Callbacks
        cbcks = [
            MlflowCallback(Phase.TRAIN_EPOCH_END, freq=1, client=mlclient, params=params),
            MlflowCallback(Phase.VALIDATION_EPOCH_END, freq=1, client=mlclient),
            SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                              freq=5,
                                              batch_idxs=[0, len(seg_trainer.train_loader) - 1],
                                              last_img_idx_in_batch=4,
                                              num_classes=len(dataset.classes),
                                              undo_preprocessing=dataset.undo_preprocess),
            *early_stop
        ]
        train_params["phase_callbacks"] = cbcks

        seg_trainer.train(train_params)
        if seg_trainer.train_loader.num_workers > 0:
            seg_trainer.train_loader._iterator._shutdown_workers()
            seg_trainer.valid_loader._iterator._shutdown_workers()
    gc.collect()
    if phase == 'train' or phase == 'test':
        if phase == 'test':
            # To make the test work, we need to set train_params anyway
            seg_trainer.init_train_params(train_params)

        test_metrics = seg_trainer.test(**test_params)
        if seg_trainer.test_loader.num_workers > 0:
            seg_trainer.test_loader._iterator._shutdown_workers()

        # log test metrics
        mlclient.log_metrics(test_metrics)

    if phase == 'run':
        seg_trainer.init_train_params(train_params)
        cbcks = [
            SaveSegmentationPredictionsCallback(phase=Phase.POST_TRAINING,
                                                path="predictions",
                                                num_classes=len(seg_trainer.test_loader.dataset.classes),
                                                )
            ]
        seg_trainer.test_loader.dataset.return_name = True
        seg_trainer.run(seg_trainer.test_loader, callbacks=cbcks)


if __name__ == '__main__':
    param_path = 'parameters.yaml'
    with open(param_path, 'r') as param_stream:
        settings = YAML().load(param_stream)

    exp_settings = settings['experiment']
    grids = settings['parameters']

    logger.info(f'Loaded parameters from {param_path}')
    runs = make_grid(grids)
    logger.info(f'Found {len(runs)} experiments')

    continue_with_errors = exp_settings.pop('continue_with_errors')

    for i, params in enumerate(runs):
        try:
            logger.info(f'Running experiment {i + 1} out of {len(runs)}')
            run({**exp_settings, **params})
            gc.collect()
        except Exception as e:
            logger.error(f'Experiment {i + 1} failed with error {e}')
            if not continue_with_errors:
                raise e
