import os
from typing import Union, Callable, Mapping, Any, List

import wandb
import numpy as np
import torch
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.utils import AverageMeter
from PIL import ImageColor, Image


class SegmentationVisualizationCallback(PhaseCallback):
    """
    A callback that adds a visualization of a batch of segmentation predictions to context.sg_logger
    Attributes:
        freq: frequency (in epochs) to perform this callback.
        batch_idx: batch index to perform visualization for.
        last_img_idx_in_batch: Last image index to add to log. (default=-1, will take entire batch).
    """

    def __init__(self, phase: Phase, freq: int, num_classes, batch_idxs=None, last_img_idx_in_batch: int = None,
                 undo_preprocessing=None):
        super(SegmentationVisualizationCallback, self).__init__(phase)
        if batch_idxs is None:
            batch_idxs = [0]
        self.freq = freq
        self.num_classes = num_classes
        self.batch_idxs = batch_idxs
        self.last_img_idx_in_batch = last_img_idx_in_batch
        self.undo_preprocesing = undo_preprocessing
        self.prefix = 'train' if phase == Phase.TRAIN_EPOCH_END else 'val' \
            if phase == Phase.VALIDATION_BATCH_END else 'test'
        if phase == Phase.TEST_BATCH_END:
            self.table = wandb.Table(columns=['ID', 'Image'])
        else:
            self.table = None

    def __call__(self, context: PhaseContext):
        epoch = context.epoch if context.epoch is not None else 0
        if epoch % self.freq == 0 and context.batch_idx in self.batch_idxs:
            preds = context.preds.clone()
            SegmentationVisualization.visualize_batch(context.inputs,
                                                      preds, context.target, self.num_classes,
                                                      context.batch_idx,
                                                      undo_preprocessing_func=self.undo_preprocesing,
                                                      prefix=self.prefix,
                                                      table=self.table,
                                                      names=context.input_name)
            if self.prefix == 'test' and context.batch_idx == self.batch_idxs[-1]:
                wandb.log({f"{self.prefix}_seg": self.table})


class SegmentationVisualization:

    @staticmethod
    def _visualize_image(image_np: np.ndarray, pred_mask: torch.Tensor, target_mask: torch.Tensor, classes):
        pred_mask = torch.tensor(pred_mask.copy())
        target_mask = torch.tensor(target_mask.copy())

        pred_mask = pred_mask.argmax(dim=0)
        target_mask = target_mask.argmax(dim=0)

        if image_np.shape[0] < 3:
            image_np = torch.vstack([image_np,
                                     torch.zeros((3 - image_np.shape[0], *image_np.shape[1:]), dtype=torch.uint8)]
                                    )
        image_np = image_np[:3, :, :]  # Take only 3 bands if there are more

        img = wandb.Image(np.moveaxis(image_np.numpy(), 0, -1), masks={
            "predictions": {
                "mask_data": pred_mask.numpy(),
                "class_labels": classes
            },
            "ground_truth": {
                "mask_data": target_mask.numpy(),
                "class_labels": classes
            }
        })
        return img

    @staticmethod
    def visualize_batch(image_tensor: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, num_classes,
                        batch_name: Union[int, str],
                        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = lambda x: x,
                        image_scale: float = 1., prefix: str = '', table: wandb.Table = None,
                        names: List[str] = None):
        """
        A helper function to visualize detections predicted by a network:
        saves images into a given path with a name that is {batch_name}_{imade_idx_in_the_batch}.jpg, one batch per call.
        Colors are generated on the fly: uniformly sampled from color wheel to support all given classes.

        :param image_tensor:            rgb images, (B, H, W, 3)
        :param pred_boxes:              boxes after NMS for each image in a batch, each (Num_boxes, 6),
                                        values on dim 1 are: x1, y1, x2, y2, confidence, class
        :param target_boxes:            (Num_targets, 6), values on dim 1 are: image id in a batch, class, x y w h
                                        (coordinates scaled to [0, 1])
        :param batch_name:              id of the current batch to use for image naming

        :param checkpoint_dir:          a path where images with boxes will be saved. if None, the result images will
                                        be returns as a list of numpy image arrays

        :param undo_preprocessing_func: a function to convert preprocessed images tensor into a batch of cv2-like images
        :param image_scale:             scale factor for output image
        """
        image_np = undo_preprocessing_func(image_tensor.detach()).type(dtype=torch.uint8).cpu()

        if names is None:
            names = ['_'.join([prefix, 'seg', str(batch_name), str(i)]) if prefix == 'val' else \
                              '_'.join([prefix, 'seg', str(batch_name * image_np.shape[0] + i)]) for i in
                          range(image_np.shape[0])]

        for i in range(image_np.shape[0]):
            preds = pred_mask[i].detach().cpu().numpy()
            targets = target_mask[i].detach().cpu().numpy()

            img = SegmentationVisualization._visualize_image(image_np[i], preds, targets, num_classes)
            if prefix == 'val':
                wandb.log({names[i]: img})
            else:
                table.add_data(names[i], img)


# class MlflowCallback(PhaseCallback):
#     """
#     A callback that logs metrics to MLFlow.
#     """
#
#     def __init__(self, phase: Phase, freq: int,
#                  client: MLRun,
#                  params: Mapping = None
#                  ):
#         """
#         param phase: phase to log metrics for
#         param freq: frequency of logging
#         param client: MLFlow client
#         """
#
#         if phase == Phase.TRAIN_EPOCH_END:
#             self.prefix = 'train_'
#         elif phase == Phase.VALIDATION_EPOCH_END:
#             self.prefix = 'val_'
#         else:
#             raise NotImplementedError('Unrecognized Phase')
#
#         super(MlflowCallback, self).__init__(phase)
#         self.freq = freq
#         self.client = client
#
#         if params:
#             self.client.log_params(params)
#
#     def __call__(self, context: PhaseContext):
#         """
#         Logs metrics to MLFlow.
#             param context: context of the current phase
#         """
#         if context.epoch % self.freq == 0:
#             self.client.log_metrics({self.prefix + k: v for k, v in context.metrics_dict.items()})


class WandbCallback(PhaseCallback):
    """
    A callback that logs metrics to MLFlow.
    """

    def __init__(self, phase: Phase, freq: int
                 ):
        """
        param phase: phase to log metrics for
        param freq: frequency of logging
        param client: MLFlow client
        """

        if phase == Phase.TRAIN_EPOCH_END:
            self.prefix = 'train_'
        elif phase == Phase.VALIDATION_EPOCH_END:
            self.prefix = 'val_'
        else:
            raise NotImplementedError('Unrecognized Phase')

        super(WandbCallback, self).__init__(phase)
        self.freq = freq

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        if self.phase == Phase.TRAIN_EPOCH_END:
            wandb.log({'epoch': context.epoch})
        if context.epoch % self.freq == 0:
            wandb.log({self.prefix + k: v for k, v in context.metrics_dict.items()})


class AverageMeterCallback(PhaseCallback):
    def __init__(self):
        super(AverageMeterCallback, self).__init__(Phase.TEST_BATCH_END)
        self.meters = {}

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        context.metrics_compute_fn.update(context.preds, context.target)
        metrics_dict = context.metrics_compute_fn.compute()
        for k, v in metrics_dict.items():
            if not self.meters.get(k):
                self.meters[k] = AverageMeter()
            self.meters[k].update(v, 1)


class SaveSegmentationPredictionsCallback(PhaseCallback):
    def __init__(self, phase, path, num_classes):
        super(SaveSegmentationPredictionsCallback, self).__init__(phase)
        self.path = path
        self.num_classes = num_classes

        os.makedirs(self.path, exist_ok=True)
        colors = ['blue', 'green', 'red']
        self.colors = []
        for color in colors:
            if isinstance(color, str):
                color = ImageColor.getrgb(color)
            self.colors.append(torch.tensor(color, dtype=torch.uint8))

    def __call__(self, context: PhaseContext):
        for prediction, input_name in zip(context.preds, context.input_name):
            path = os.path.join(self.path, input_name)
            prediction = prediction.detach().cpu()
            masks = torch.concat([
                (prediction.argmax(0) == cls).unsqueeze(0)
                for cls in range(self.num_classes)
            ])

            img_to_draw = torch.zeros(*prediction.shape[-2:], 3, dtype=torch.uint8)
            # TODO: There might be a way to vectorize this
            for mask, color in zip(masks, self.colors):
                img_to_draw[mask] = color

            Image.fromarray(img_to_draw.numpy()).save(path)
