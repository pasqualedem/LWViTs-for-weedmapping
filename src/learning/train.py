import os
from typing import Mapping

import mlflow
import torch

import numpy as np
from mlflow.tracking import MlflowClient

from super_gradients.training.datasets.dataset_interfaces.dataset_interface import SuperviselyPersonsDatasetInterface
from super_gradients.training import SgModel
from torchmetrics import JaccardIndex, Accuracy
from super_gradients.training.utils.callbacks import Phase
from ruamel.yaml import YAML

from callbacks import SegmentationVisualizationCallback, MlflowCallback
from models.segnet import SegNet
from data.sequoia import SequoiaDatasetInterface

from loss import LOSSES as LOSSES_DICT
from metrics import METRICS as METRICS_DICT
from utils import setup_mlflow, MLRun

torch.manual_seed(42)
np.random.seed(42)
EXP_DIR = 'mlflow'

os.environ['ENVIRONMENT_NAME'] = 'development'
LOSS_WEIGHTS = torch.tensor([0.0273, 1.0, 4.3802])
EXP_NAME = 'Try'


def parse_params(params: dict) -> (dict, dict, dict):
    # Instantiate loss
    input_train_params = params['train_params']
    loss_params = params['train_params'].pop('loss')
    loss = LOSSES_DICT[loss_params['name']](**loss_params['params'])

    # metrics
    metrics_params = params['metrics']
    metrics = [
        METRICS_DICT[name](**params)
        for name, params in metrics_params.items()
    ]

    # dataset params
    dataset_params = params['dataset']

    train_params = {
        "greater_metric_to_watch_is_better": True,
        "train_metrics_list": metrics,
        "valid_metrics_list": metrics,
        "loss_logging_items_names": ["loss"],
        "loss": loss,
        **input_train_params
    }

    test_params = {
        "test_metrics_list": metrics,
    }

    return train_params, test_params, dataset_params


def experiment(param_path: str = 'parameters.yaml'):
    with open(param_path, 'r') as param_stream:
        params = YAML().load(param_stream)

    train_params, test_params, dataset_params = parse_params(params)

    # Mlflow
    mlclient = MLRun(EXP_NAME)

    sg_model = SgModel(experiment_name='SegNetTry3', ckpt_root_dir=mlclient.run.info.artifact_uri)
    dataset = SequoiaDatasetInterface(dataset_params, channels='CIR')
    sg_model.connect_dataset_interface(dataset, data_loader_num_workers=0)

    # Callbacks
    cbcks = [
        MlflowCallback(Phase.TRAIN_EPOCH_END, freq=1, client=mlclient, params=train_params),
        MlflowCallback(Phase.VALIDATION_EPOCH_END, freq=1, client=mlclient),
        SegmentationVisualizationCallback(phase=Phase.VALIDATION_BATCH_END,
                                          freq=5,
                                          last_img_idx_in_batch=4,
                                          num_classes=len(dataset.classes),
                                          undo_preprocessing=dataset.undo_preprocess),

    ]
    train_params["phase_callbacks"] = cbcks

    model = SegNet(in_chn=3, out_chn=3)
    # model = "regseg48"
    sg_model.build_model(model)
    sg_model.train(train_params)
    test_metrics = sg_model.test(**test_params)

    # log test metrics
    metric_names = params['metrics'].keys()
    mlclient.log_metrics(
        {'test_loss': test_metrics[0], **{name: value for 'test_' + name, value in zip(metric_names, test_metrics[1:])}})


if __name__ == '__main__':
    experiment()
