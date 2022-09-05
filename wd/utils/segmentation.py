import numpy as np
import torch


def tensor_to_segmentation_image(prediction: torch.Tensor, cmap) -> np.array:
    segmented_image = np.ones((*prediction.shape, 3), dtype="uint8")
    for class_value, class_color in cmap.items():
        segmented_image[prediction == class_value] = class_color
    return segmented_image

