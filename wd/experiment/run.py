import gc
import os

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training.utils.callbacks import Phase

from wd.callbacks import WandbCallback, SegmentationVisualizationCallback
from wd.data.sequoia import WeedMapDatasetInterface
from wd.experiment.parameters import parse_params
from wd.learning.seg_trainer import SegmentationTrainer
from wd.utils.utilities import dict_to_yaml_string, values_to_number, nested_dict_update

import wd.data

logger = get_logger(__name__)


class Run:
    def __init__(self):
        self.params = None
        self.dataset = None
        self.early_stop = None
        self.dataset_params = None
        self.seg_trainer = None
        self.train_params = None
        self.test_params = None
        self.run_params = None
        self.phases = None

    def init(self, params: dict):
        self.seg_trainer = None
        try:
            self.params = params
            self.phases = params['phases']

            self.train_params, self.test_params, self.dataset_params, self.early_stop = parse_params(params)
            self.run_params = params.get('run_params') or {}

            self.seg_trainer = SegmentationTrainer(experiment_name=params['experiment']['group'],
                                                   ckpt_root_dir=params['experiment']['tracking_dir']
                                                   if params['experiment']['tracking_dir'] else 'wandb')
            dataset_interface = self._get_dataset_interface()
            self.dataset = dataset_interface(self.dataset_params)
            self.seg_trainer.connect_dataset_interface(self.dataset,
                                                       data_loader_num_workers=params['dataset']['num_workers'])
            self.seg_trainer.init_model(params, False, None)
            self.seg_trainer.init_loggers({"in_params": params}, self.train_params)
            logger.info(f"Input params: \n\n {dict_to_yaml_string(params)}")
        except Exception as e:
            if self.seg_trainer is not None:
                self.seg_trainer.sg_logger.close(True)
            raise e

    def resume(self, wandb_run, updated_config, phases):
        try:
            try:
                self.params = values_to_number(wandb_run.config['in_params'])
            except KeyError:
                raise RuntimeError("No params recorded for run, just delete it!")
            self.params = nested_dict_update(self.params, updated_config)
            self.phases = phases
            wandb_run.config['in_params'] = self.params
            wandb_run.update()
            self.train_params, self.test_params, self.dataset_params, self.early_stop = parse_params(self.params)

            self.seg_trainer = SegmentationTrainer(experiment_name=self.params['experiment']['group'],
                                              ckpt_root_dir=self.params['experiment']['tracking_dir']
                                              if self.params['experiment']['tracking_dir'] else 'wandb')
            dataset_interface = self._get_dataset_interface()
            self.dataset = dataset_interface(self.dataset_params)
            self.seg_trainer.connect_dataset_interface(self.dataset, data_loader_num_workers=self.params['dataset']['num_workers'])
            track_dir = wandb_run.config.get('in_params').get('experiment').get('tracking_dir') or 'wandb'
            checkpoint_path_group = os.path.join(track_dir, wandb_run.group, 'wandb')
            run_folder = list(filter(lambda x: str(wandb_run.id) in x, os.listdir(checkpoint_path_group)))
            checkpoint_path = None
            if 'epoch' in wandb_run.summary:
                ckpt = 'ckpt_latest.pth' if 'train' in phases else 'ckpt_best.pth'
                checkpoint_path = os.path.join(checkpoint_path_group, run_folder[0], 'files', ckpt)
            self.seg_trainer.init_model(self.params, True, checkpoint_path)
            self.seg_trainer.init_loggers({"in_params": self.params}, self.train_params, run_id=wandb_run.id)
        except Exception as e:
            if self.seg_trainer is not None:
                self.seg_trainer.sg_logger.close(really=True)
            raise e

    def _get_dataset_interface(self):
        try:
            dataset_interface = getattr(wd.data, self.params['dataset_interface'])
        except AttributeError:
            raise AttributeError("No interface found!")
        return dataset_interface

    def launch(self):
        try:
            if 'train' in self.phases:
                train(self.seg_trainer, self.train_params, self.dataset, self.early_stop)

            if 'test' in self.phases:
                test_metrics = self.seg_trainer.test(**self.test_params)

            if 'inference' in self.phases:
                inference(self.seg_trainer, self.run_params, self.dataset)
        finally:
            if self.seg_trainer is not None:
                self.seg_trainer.sg_logger.close(True)


def train(seg_trainer, train_params, dataset, early_stop):
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
