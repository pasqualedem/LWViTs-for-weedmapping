import os
import numbers
from typing import Iterable

import PIL
import torch
from torch import Tensor
from torch.nn.functional import one_hot

from torchvision.transforms import functional as F, InterpolationMode

from PIL import ImageOps


class PairSystematicTransform:

    def __init__(self, init_state, periodicity=1):
        self.periodicity = periodicity
        self.init_state = init_state

        self.pair_state = {}
        self.period_state = {}

    def __call__(self, img):
        pid = os.getpid()
        if pid in self.pair_state:
            state = self.pair_state.pop(pid)
        else:
            state, counter = self.period_state.get(pid) or (self.init_state, 0)
            self.pair_state[pid] = state
            if counter + 1 >= self.periodicity:
                self.period_state[pid] = self.change_state(state), 0
            else:
                self.period_state[pid] = state, counter + 1
        return self.transform(img, state)

    def transform(self, img, state):
        raise NotImplementedError

    def change_state(self, state):
        raise NotImplementedError


class PairFlip(PairSystematicTransform):
    def __init__(self, periodicity=1, orientation='horizontal'):
        super().__init__(False, periodicity)
        if orientation == 'horizontal':
            self.flip = F.hflip
        elif orientation == 'vertical':
            self.flip = F.vflip
        else:
            raise ValueError(f'Unknown orientation: {orientation}')

    def transform(self, img, state):
        if state:
            return self.flip(img)
        else:
            return img

    def change_state(self, state):
        return not state


class PairFourCrop(PairSystematicTransform):
    def __init__(self, size, padding=0, periodicity=1):
        super().__init__(0, periodicity)
        self.padding = padding
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def transform(self, img, state):
        if isinstance(img, PIL.Image.Image):
            h, w = img.size
        elif isinstance(img, torch.Tensor):
            w, h = img.size()[-2:]
        else:
            raise TypeError("img should be PIL.Image or torch.Tensor, not {}".format(type(img)))
        if state == 0:
            return F.crop(img, 0, 0, *self.size)
        elif state == 1:
            return F.crop(img, 0, h - self.size[1], *self.size)
        elif state == 2:
            return F.crop(img, w - self.size[0], 0, *self.size)
        elif state == 3:
            return F.crop(img, w - self.size[0], h - self.size[1], *self.size)

    def change_state(self, state):
        return (state + 1) % 4


class PairRandomCrop:
    """Crop the given PIL.Image at a random location.
    ** This is a MODIFIED version **, which supports identical random crop for
    both image and target map in Semantic Segmentation.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """
    image_crop_position = {}

    def __init__(self, size, padding=0):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        if isinstance(img, PIL.Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            w, h = img.size()[-2:]
        else:
            raise TypeError("img should be PIL.Image or torch.Tensor, not {}".format(type(img)))

        th, tw = self.size
        if w == tw and h == th:
            return img

        pid = os.getpid()
        if pid in self.image_crop_position:
            x1, y1 = self.image_crop_position.pop(pid)
        else:
            x1 = torch.randint(size=(1,), low=0, high=w - tw)
            y1 = torch.randint(size=(1,), low=0, high=w - th)
            self.image_crop_position[pid] = (x1, y1)
        return F.crop(img, x1, y1, tw, th)


class PairRandomFlip(torch.nn.Module):
    """Flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5, orientation='horizontal'):
        super().__init__()
        # _log_api_usage_once(self)
        self.p = p
        if orientation == 'horizontal':
            self.flip = F.hflip
        elif orientation == 'vertical':
            self.flip = F.vflip
        else:
            raise ValueError(f'Unknown orientation: {orientation}')
        self.image_flip = {}

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        pid = os.getpid()
        if pid in self.image_flip:
            value = self.image_flip.pop(pid)
        else:
            value = torch.rand(1)
            self.image_flip[pid] = value
        if value < self.p:
            return self.flip(img)
        return img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class PairRandomRotation(torch.nn.Module):
    """Rotate the given image randomly with a given rotation.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        degree: degree of rotation.
        interpolation: Default: InterpolationMode.BILINEAR
    """

    def __init__(self, degree, interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        # _log_api_usage_once(self)

        if isinstance(self.degree, Iterable):
            self.degree = degree
        else:
            self.degree = [-degree, degree]
        self.interpolation = interpolation
        self.image_rotation = {}

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated.

        Returns:
            PIL Image or Tensor: Randomly rotated image.
        """
        pid = os.getpid()
        if pid in self.image_rotation:
            value = self.image_rotation.pop(pid)
        else:
            value = torch.FloatTensor(1).uniform_(*self.degree)
            self.image_rotation[pid] = value

        return F.rotate(img, value, self.interpolation)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.degree})"


class ToLong:
    def __call__(self, x):
        return x.long()


class SegOneHot:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, x):
        y = torch.moveaxis(one_hot(x, self.num_classes), 2, 0)
        return y


class FixValue:
    def __init__(self, source, target):
        self.s = source
        self.t = target

    def __call__(self, x):
        x[x == self.s] = self.t
        return x


class Denormalize(torch.nn.Module):
    """Denormalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        # _log_api_usage_once(self)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: Tensor) -> Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        return tensor.mul(std.view(-1, 1, 1)).add(mean.view(-1, 1, 1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


def squeeze0(x):
    return torch.squeeze(x, dim=0)
