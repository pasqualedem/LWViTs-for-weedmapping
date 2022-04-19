import os
from typing import Union, Callable, Mapping, Any

import cv2
import numpy as np
import torch
from super_gradients.training.utils.callbacks import PhaseCallback, Phase, PhaseContext
from super_gradients.training.utils.utils import AverageMeter
from torchvision.utils import draw_segmentation_masks
from PIL import ImageColor, Image

from wd.utils.utils import MLRun


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

    def __call__(self, context: PhaseContext):
        epoch = context.epoch if context.epoch is not None else 0
        if epoch % self.freq == 0 and context.batch_idx in self.batch_idxs:
            preds = context.preds.clone()
            batch_imgs = SegmentationVisualization.visualize_batch(context.inputs,
                                                                   preds, context.target, self.num_classes,
                                                                   context.batch_idx,
                                                                   undo_preprocessing_func=self.undo_preprocesing)
            batch_imgs = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in batch_imgs]
            batch_imgs = np.stack(batch_imgs)
            tag = self.prefix + "_batch_" + str(context.batch_idx) + "_images"
            tag = tag + f"_epoch_{context.epoch}" if context.epoch is not None else tag
            context.sg_logger.add_images(tag=tag, images=batch_imgs[:self.last_img_idx_in_batch],
                                         global_step=epoch, data_format='NHWC')


class SegmentationVisualization:

    @staticmethod
    def _visualize_image(image_np: np.ndarray, pred_mask: torch.Tensor, target_mask: torch.Tensor, num_classes,
                         image_scale: float, checkpoint_dir: str, image_name: str):
        pred_mask = torch.tensor(pred_mask.copy())
        # image_np = image_np.moveaxis(0, -1)
        #
        # pred_mask = pred_mask[np.newaxis, :, :] > 0.5
        # target_mask = target_mask[np.newaxis, :, :].astype(bool)
        # tp_mask = np.logical_and(pred_mask, target_mask)
        # fp_mask = np.logical_and(pred_mask, np.logical_not(target_mask))
        # fn_mask = np.logical_and(np.logical_not(pred_mask), target_mask)
        overlay = torch.concat([
            (pred_mask.argmax(0) == cls).unsqueeze(0)
            for cls in range(num_classes)
        ])
        # overlay = torch.from_numpy(np.concatenate([tp_mask, fp_mask, fn_mask]))

        # SWITCH BETWEEN BLUE AND RED IF WE SAVE THE IMAGE ON THE DISC AS OTHERWISE WE CHANGE CHANNEL ORDERING
        colors = ['green', 'red', 'blue']
        image_np = image_np[:3, :, :]  # Take only 3 bands if there are more
        if image_np.shape[0] < 3:
            image_np = torch.vstack([image_np,
                                     torch.zeros((3 - image_np.shape[0], *image_np.shape[1:]), dtype=torch.uint8)])

        res_image = draw_segmentation_masks(image_np, overlay, colors=colors, alpha=0.4).detach().numpy()
        res_image = np.concatenate([res_image[ch, :, :, np.newaxis] for ch in range(3)], 2)
        res_image = cv2.resize(res_image.astype(np.uint8), (0, 0), fx=image_scale, fy=image_scale,
                               interpolation=cv2.INTER_NEAREST)

        if checkpoint_dir is None:
            return res_image
        else:
            cv2.imwrite(os.path.join(checkpoint_dir, str(image_name) + '.jpg'), res_image)

    @staticmethod
    def visualize_batch(image_tensor: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, num_classes,
                        batch_name: Union[int, str], checkpoint_dir: str = None,
                        undo_preprocessing_func: Callable[[torch.Tensor], np.ndarray] = lambda x: x,
                        image_scale: float = 1.):
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
        image_np = undo_preprocessing_func(image_tensor.detach().cpu())
        # pred_mask = torch.sigmoid(pred_mask[:, 0, :, :])  # comment out

        out_images = []
        for i in range(image_np.shape[0]):
            preds = pred_mask[i].detach().cpu().numpy()
            targets = target_mask[i].detach().cpu().numpy()

            image_name = '_'.join([str(batch_name), str(i)])
            res_image = SegmentationVisualization._visualize_image(image_np[i], preds, targets, num_classes,
                                                                   image_scale, checkpoint_dir, image_name)
            if res_image is not None:
                out_images.append(res_image)

        return out_images


class MlflowCallback(PhaseCallback):
    """
    A callback that logs metrics to MLFlow.
    """

    def __init__(self, phase: Phase, freq: int,
                 client: MLRun,
                 params: Mapping = None
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

        super(MlflowCallback, self).__init__(phase)
        self.freq = freq
        self.client = client

        if params:
            self.client.log_params(params)

    def __call__(self, context: PhaseContext):
        """
        Logs metrics to MLFlow.
            param context: context of the current phase
        """
        if context.epoch % self.freq == 0:
            self.client.log_metrics({self.prefix + k: v for k, v in context.metrics_dict.items()})


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