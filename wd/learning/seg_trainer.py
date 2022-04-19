from typing import Mapping

from super_gradients.common.abstractions.abstract_logger import get_logger
from super_gradients.training import SgModel, StrictLoad
from super_gradients.training.params import TrainingParams
from super_gradients.training.utils import sg_model_utils
from super_gradients.training.utils.callbacks import Phase, PhaseContext, CallbackHandler
from super_gradients.training import utils as core_utils

import torch
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from tqdm import tqdm

from wd.callbacks import AverageMeterCallback, SegmentationVisualizationCallback
from wd.models import MODELS as MODELS_DICT
from wd.utils.utils import MLRun

logger = get_logger(__name__)


class SegmentationTrainer(SgModel):
    def init_model(self, params: Mapping, phases: list, mlflowclient: MLRun):
        # init model
        model_params = params['model']
        if model_params['name'] in MODELS_DICT.keys():
            model = MODELS_DICT[model_params['name']](**model_params['params'],
                                                      in_chn=len(params['dataset']['channels']),
                                                      out_chn=params['dataset']['num_classes']
                                                      )
        else:
            model = model_params['name']

        self.build_model(model)
        if 'train' not in phases:
            ckpt_local_path = mlflowclient.run.info.artifact_uri + '/SG/ckpt_best.pth'
            self.checkpoint = load_checkpoint_to_model(ckpt_local_path=ckpt_local_path,
                                                       load_backbone=False,
                                                       net=self.net,
                                                       strict=StrictLoad.ON.value,
                                                       load_weights_only=self.load_weights_only)

            if 'ema_net' in self.checkpoint.keys():
                logger.warning("[WARNING] Main network has been loaded from checkpoint but EMA network exists as well. It "
                               " will only be loaded during validation when training with ema=True. ")

            # UPDATE TRAINING PARAMS IF THEY EXIST & WE ARE NOT LOADING AN EXTERNAL MODEL's WEIGHTS
            self.best_metric = self.checkpoint['acc'] if 'acc' in self.checkpoint.keys() else -1
            self.start_epoch = self.checkpoint['epoch'] if 'epoch' in self.checkpoint.keys() else 0

    def train(self, training_params: dict = dict()):
        super().train(training_params)
        if self.train_loader.num_workers > 0:
            self.train_loader._iterator._shutdown_workers()
            self.valid_loader._iterator._shutdown_workers()

    def test(self,  # noqa: C901
             test_loader: torch.utils.data.DataLoader = None,
             loss: torch.nn.modules.loss._Loss = None,
             silent_mode: bool = False,
             test_metrics: Mapping = None,
             loss_logging_items_names=None, metrics_progress_verbose=False, test_phase_callbacks=(),
             use_ema_net=True) -> dict:
        """
        Evaluates the model on given dataloader and metrics.

        :param test_loader: dataloader to perform test on.
        :param test_metrics: dict name: Metric for evaluation.
        :param silent_mode: (bool) controls verbosity
        :param metrics_progress_verbose: (bool) controls the verbosity of metrics progress (default=False). Slows down the program.
        :param use_ema_net (bool) whether to perform test on self.ema_model.ema (when self.ema_model.ema exists,
            otherwise self.net will be tested) (default=True)
        :return: results tuple (tuple) containing the loss items and metric values.

        All of the above args will override SgModel's corresponding attribute when not equal to None. Then evaluation
         is ran on self.test_loader with self.test_metrics.
        """

        test_phase_callbacks = list(test_phase_callbacks) + [
            SegmentationVisualizationCallback(phase=Phase.TEST_BATCH_END,
                                              freq=1,
                                              batch_idxs=list(range(15)),
                                              num_classes=len(self.dataset_interface.classes),
                                              undo_preprocessing=self.dataset_interface.undo_preprocess)
        ]
        metrics_values = super().test(test_loader, loss, silent_mode, list(test_metrics.values()),
                                      loss_logging_items_names,
                                      metrics_progress_verbose, test_phase_callbacks, use_ema_net)

        metric_names = test_metrics.keys()

        if self.test_loader.num_workers > 0:
            self.test_loader._iterator._shutdown_workers()
        return {'test_loss': metrics_values[0], **dict(zip(metric_names, metrics_values[1:]))}

    def init_train_params(self, train_params: Mapping = None, init_sg_loggers: bool = True) -> None:
        if self.training_params is None:
            self.training_params = TrainingParams()
        self.training_params.override(**train_params)
        if init_sg_loggers:
            self._initialize_sg_logger_objects()
        if self.phase_callbacks is None:
            self.phase_callbacks = []
        self.phase_callback_handler = CallbackHandler(self.phase_callbacks)

    def run(self, data_loader: torch.utils.data.DataLoader, callbacks=None, silent_mode: bool = False):
        """
        Runs the model on given dataloader.

        :param data_loader: dataloader to perform run on

        """

        # THE DISABLE FLAG CONTROLS WHETHER THE PROGRESS BAR IS SILENT OR PRINTS THE LOGS
        if callbacks is None:
            callbacks = []
        progress_bar_data_loader = tqdm(data_loader, bar_format="{l_bar}{bar:10}{r_bar}", dynamic_ncols=True,
                                        disable=silent_mode)
        context = PhaseContext(criterion=self.criterion,
                               device=self.device,
                               sg_logger=self.sg_logger)

        self.phase_callbacks.extend(callbacks)

        if not silent_mode:
            # PRINT TITLES
            pbar_start_msg = f"Running model on {len(data_loader)} batches"
            progress_bar_data_loader.set_description(pbar_start_msg)

        with torch.no_grad():
            for batch_idx, batch_items in enumerate(progress_bar_data_loader):
                batch_items = core_utils.tensor_container_to_device(batch_items, self.device, non_blocking=True)

                additional_batch_items = {}
                targets = None
                if hasattr(batch_items, '__len__'):
                    if len(batch_items) == 2:
                        inputs, targets = batch_items
                    elif len(batch_items) == 3:
                        inputs, targets, additional_batch_items = batch_items
                    else:
                        raise ValueError(f"Expected 1, 2 or 3 items in batch_items, got {len(batch_items)}")
                else:
                    inputs = batch_items

                output = self.net(inputs)

                context.update_context(batch_idx=batch_idx,
                                       inputs=inputs,
                                       preds=output,
                                       target=targets,
                                       **additional_batch_items)

                # TRIGGER PHASE CALLBACKS
                self.phase_callback_handler(Phase.POST_TRAINING, context)
