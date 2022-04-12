from typing import Mapping

from super_gradients.training import SgModel
from super_gradients.training.utils.callbacks import Phase

import torch

from callbacks import AverageMeterCallback, SegmentationVisualizationCallback


class Trainer(SgModel):
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
            AverageMeterCallback(),
            SegmentationVisualizationCallback(phase=Phase.TEST_BATCH_END,
                                              freq=1,
                                              batch_idxs=list(range(50)),
                                              num_classes=len(self.dataset_interface.classes),
                                              undo_preprocessing=self.dataset_interface.undo_preprocess)
        ]
        metrics_values = super().test(test_loader, loss, silent_mode, list(test_metrics.values()),
                                      loss_logging_items_names,
                                      metrics_progress_verbose, test_phase_callbacks, use_ema_net)

        metric_names = test_metrics.keys()
        return {'test_loss': metrics_values[0], **dict(zip(metric_names, metrics_values[1:]))}

