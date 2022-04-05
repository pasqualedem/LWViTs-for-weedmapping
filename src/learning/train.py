import os
from typing import Any, Mapping

import torch

import numpy as np

# os.environ['ENVIRONMENT_NAME'] = 'development'

from super_gradients.training import SgModel
from super_gradients.training.utils.early_stopping import EarlyStop
from super_gradients.training.utils.callbacks import Phase
from super_gradients.common.abstractions.abstract_logger import get_logger
from sklearn.model_selection import ParameterGrid
from ruamel.yaml import YAML

from callbacks import SegmentationVisualizationCallback, MlflowCallback
from data.sequoia import SequoiaDatasetInterface

from loss import LOSSES as LOSSES_DICT
from metrics import METRICS as METRICS_DICT
from models import MODELS as MODELS_DICT
from utils.utils import MLRun
from utils.grid import make_grid

torch.manual_seed(42)
np.random.seed(42)
EXP_DIR = 'mlflow'
logger = get_logger(__name__)


def parse_params(params: dict) -> (dict, dict, dict, Any, list):
    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params'].pop('loss')
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    train_metrics_params = params['train_metrics']
    train_metrics = [
        METRICS_DICT[name](**params)
        for name, params in train_metrics_params.items()
    ]

    test_metrics_params = params['train_metrics']
    test_metrics = [
        METRICS_DICT[name](**params)
        for name, params in test_metrics_params.items()
    ]

    # init model
    model_params = params['model']
    if model_params['name'] in MODELS_DICT.keys():
        model = MODELS_DICT[model_params['name']](**model_params['params'],
                                                  in_chn=len(params['dataset']['channels']),
                                                  out_chn=params['dataset']['num_classes']
                                                  )
    else:
        model = model_params['name']

    # dataset params
    dataset_params = params['dataset']

    train_params = {
        "greater_metric_to_watch_is_better": True,
        "train_metrics_list": train_metrics,
        "valid_metrics_list": test_metrics,
        "loss_logging_items_names": ["loss"],
        "loss": loss,
        **input_train_params
    }

    test_params = {
        "test_metrics_list": test_metrics,
    }

    # early stopping
    early_stopping_params = params['early_stopping']
    if early_stopping_params['enabled']:
        early_stop = [EarlyStop(Phase.VALIDATION_EPOCH_END, **early_stopping_params['params'])]
    else:
        early_stop = []

    return train_params, test_params, dataset_params, model, early_stop


def experiment(params: Mapping):

    exp_name = params['experiment']
    description = params['description']
    params = params['parameters']
    train_params, test_params, dataset_params, model, early_stop = parse_params(params)

    # Mlflow
    mlclient = MLRun(exp_name, description)

    sg_model = SgModel(experiment_name='SegNetTry3', ckpt_root_dir=mlclient.run.info.artifact_uri)
    dataset = SequoiaDatasetInterface(dataset_params)
    sg_model.connect_dataset_interface(dataset, data_loader_num_workers=params['dataset']['num_workers'])

    # Callbacks
    cbcks = [
        MlflowCallback(Phase.TRAIN_EPOCH_END, freq=1, client=mlclient, params=train_params),
        MlflowCallback(Phase.VALIDATION_EPOCH_END, freq=1, client=mlclient),
        SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                          freq=5,
                                          last_img_idx_in_batch=4,
                                          num_classes=len(dataset.classes),
                                          undo_preprocessing=dataset.undo_preprocess),
        *early_stop
    ]
    train_params["phase_callbacks"] = cbcks

    sg_model.build_model(model)
    sg_model.train(train_params)
    test_metrics = sg_model.test(**test_params)

    # log test metrics
    metric_names = params['test_metrics'].keys()
    mlclient.log_metrics(
        {'test_loss': test_metrics[0], **{'test_' + name: value
                                          for name, value in zip(metric_names, test_metrics[1:])}})


if __name__ == '__main__':
    param_path = 'parameters.yaml'
    with open(param_path, 'r') as param_stream:
        grids = YAML().load(param_stream)

    logger.info(f'Loaded parameters from {param_path}')
    experiments = make_grid(grids)
    logger.info(f'Found {len(experiments)} experiments')

    for i, params in enumerate(experiments):
        logger.info(f'Running experiment {i} out of {len(experiments)}')
        experiment(params)
