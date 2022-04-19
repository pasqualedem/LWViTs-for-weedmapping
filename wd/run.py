import os
from typing import Any, Mapping

import torch
import gc

import numpy as np

from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from ruamel.yaml import YAML

from wd.callbacks import SegmentationVisualizationCallback, MlflowCallback, SaveSegmentationPredictionsCallback
from wd.data.sequoia import SequoiaDatasetInterface
from wd.loss import LOSSES as LOSSES_DICT
from wd.metrics import metrics_factory
from wd.utils.utils import MLRun, mlflow_server
from wd.utils.grid import make_grid
from wd.learning.seg_trainer import SegmentationTrainer

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
    phases = params['phases']

    train_params, test_params, dataset_params, early_stop = parse_params(params)

    # Mlflow
    if 'train' not in phases:
        run_hash = params['run_hash']
    else:
        run_hash = None
    mlclient = MLRun(exp_name, description, run_hash)

    seg_trainer = SegmentationTrainer(experiment_name='SG', ckpt_root_dir=mlclient.run.info.artifact_uri)
    dataset = SequoiaDatasetInterface(dataset_params)
    seg_trainer.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])
    seg_trainer.init_model(params, phases, mlclient)

    if 'train' in phases:  # ------------------------ TRAINING PHASE ------------------------
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

    gc.collect()
    if 'test' in phases:  # ------------------------ TEST PHASE ------------------------
        if 'train' not in phases:
            # To make the test work, we need to set train_params anyway
            seg_trainer.init_train_params(train_params, params['test_params']['init_sg_loggers'])

        test_metrics = seg_trainer.test(**test_params)

        # log test metrics
        mlclient.log_metrics(test_metrics)

    if 'run' in phases:  # ------------------------ RUN PHASE ------------------------
        run_params = params['run_params']
        run_loader = dataset.get_run_loader(folders=run_params['run_folders'], batch_size=run_params['batch_size'])
        seg_trainer.init_train_params(train_params, run_params['init_sg_loggers'])
        cbcks = [
            SaveSegmentationPredictionsCallback(phase=Phase.POST_TRAINING,
                                                path=
                                                run_params['prediction_folder']
                                                if run_params['prediction_folder'] != 'mlflow'
                                                else mlclient.run.info.artifact_uri + '/predictions',
                                                num_classes=len(seg_trainer.test_loader.dataset.classes),
                                                )
        ]
        run_loader.dataset.return_name = True
        seg_trainer.run(run_loader, callbacks=cbcks)
        # seg_trainer.valid_loader.dataset.return_name = True
        # seg_trainer.run(seg_trainer.valid_loader, callbacks=cbcks)


def experiment(settings: Mapping, param_path: str = "local variable"):
    exp_settings = settings['experiment']
    grids = settings['parameters']

    mlflow_server(exp_settings['mlruns_folder'])
    logger.info('Server started!')

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


if __name__ == '__main__':
    param_path = 'parameters.yaml'
    with open(param_path, 'r') as param_stream:
        settings = YAML().load(param_stream)

    experiment(settings)

