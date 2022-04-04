import os
from typing import Any, Union, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface
from transforms import PairRandomCrop, SegOneHot, ToLong, FixValue, Denormalize
from sklearn.model_selection import train_test_split


class SequoiaDatasetInterface(DatasetInterface):
    MEAN_STDS = \
        {'CIR':
            (
                (0.2927, 0.3166, 0.3368),
                (0.1507, 0.1799, 0.1966)
            )
        }

    def __init__(self, dataset_params, name="sequoia", channels='CIR'):
        super(SequoiaDatasetInterface, self).__init__(dataset_params)
        self.dataset_name = name
        self.channels = channels
        self.lib_dataset_params = {'mean': self.MEAN_STDS[channels][0], 'std': self.MEAN_STDS[channels][1]}

        crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=320)

        transform_train = transforms.Compose([
            PairRandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        transform_test = transforms.Compose([
            PairRandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        target_transform = transforms.Compose([
            PairRandomCrop(crop_size),
            transforms.PILToTensor(),
            lambda x: torch.squeeze(x, dim=0),
            ToLong(),
            FixValue(source=10000, target=1),
            SegOneHot(num_classes=len(SequoiaDataset.CLASS_LABELS.keys()))
        ])

        # Divide train, val and test
        train_folders = ['006', '007']  # Fixed subfolders
        test_folders = ['005']  # Fixed subfolders

        train_index = SequoiaDataset.build_index(self.dataset_params.root, train_folders, channels)
        test_index = SequoiaDataset.build_index(self.dataset_params.root, test_folders, channels)
        train_index, val_index = train_test_split(train_index, test_size=0.2)

        self.trainset = SequoiaDataset(root=self.dataset_params.root, train=True,
                                       batch_size=self.dataset_params.batch_size, index=train_index,
                                       transform=transform_train, target_transform=target_transform, channels=channels)

        self.valset = SequoiaDataset(root=self.dataset_params.root, train=False,
                                     batch_size=self.dataset_params.val_batch_size, index=val_index,
                                     transform=transform_test, target_transform=target_transform, channels=channels)

        self.testset = SequoiaDataset(root=self.dataset_params.root, train=False,
                                      batch_size=self.dataset_params.test_batch_size, index=test_index,
                                      transform=transform_test, target_transform=target_transform, channels=channels)

    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)


class SequoiaDataset(VisionDataset):
    CLASS_LABELS = {0: "background", 1: "crop", 2: 'weed'}
    classes = ['background', 'crop', 'weed']

    def __init__(self,
                 root: str,
                 transform: callable,
                 target_transform: callable,
                 channels: Union[str, Iterable] = 'CIR',
                 index: Iterable = None,
                 batch_size: int = 4,
                 train: bool = True,
                 return_mask: bool = False
                 ):
        """
        Initialize a sequence of Graph User-Item IDs.

        :param batch_size: The batch size.
        """
        self.batch_size = batch_size
        super().__init__(root=root)

        if channels == 'CIR':
            self.get_img = self.get_cir
        else:
            self.get_img = self.get_channels

        if index:
            self.index = index
        else:
            self.index = self.build_index(root, channels)

        self.len = len(self.index)

        self.channels = channels
        self.n_files = {}
        self.path = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_mask = return_mask

    @classmethod
    def build_index(cls,
                    root: str,
                    macro_folders: Iterable = None,
                    channels: Union[Iterable, str] = 'CIR') -> list:

        if macro_folders is None:
            macro_folders = os.listdir(root)

        if channels == 'CIR':
            counter_channel = channels
        else:
            counter_channel = channels[0]  # Channel used to count images

        n_files = dict()
        for folder in macro_folders:
            n_files[folder] = (os.listdir(os.path.join(root, folder, 'tile', counter_channel)))
        index = dict()
        i = 0
        for folder, files in n_files.items():
            for file in files:
                index[i] = folder, file
                i += 1
        return list(index.values())

    def get_cir(self, folder, file) -> Any:
        img = Image.open(
            os.path.join(self.path, folder, 'tile', 'cir', file)
        )
        return img

    def get_channels(self, folder, file) -> Any:
        raise NotImplementedError
        imgs = []
        for c in self.channels:
            img = Image.open(
                os.path.join(self.path, folder, 'tile', c, file)
            )
            imgs.append(np.array(img))
        return np.array(imgs)

    def __getitem__(self, index: int) -> Any:
        """
        Get the i-th image and related target

        :param idx: The index of the image.
        :return: A pair consisting of the image and the target
        """
        folder, file = self.index[index]
        img = self.get_img(folder, file)
        gt = Image.open(
            os.path.join(self.path, folder, 'groundtruth',
                         folder + '_' + file.split('.')[0] + '_GroundTruth_iMap.png'
                         )
        )
        if self.return_mask:
            mask = Image.open(
                os.path.join(self.path, folder, 'mask', file)
            )
            return self.transform(img), (self.target_transform(gt), mask)
        else:
            return self.transform(img), self.target_transform(gt)

    def __len__(self) -> int:
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return self.len
