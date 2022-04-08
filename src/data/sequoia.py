import os
from typing import Any, Union, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from torchvision.datasets.vision import VisionDataset
import torchvision.transforms as transforms
from super_gradients.training.datasets.dataset_interfaces import DatasetInterface
from transforms import PairRandomCrop, SegOneHot, ToLong, FixValue, Denormalize, PairRandomFlip, squeeze0
from sklearn.model_selection import train_test_split

from torch.utils.data.distributed import DistributedSampler
from super_gradients.training import utils as core_utils
from super_gradients.training.datasets.mixup import CollateMixup
from super_gradients.training.exceptions.dataset_exceptions import IllegalDatasetParameterException
from super_gradients.common.abstractions.abstract_logger import get_logger

logger = get_logger(__name__)


class SequoiaDatasetInterface(DatasetInterface):
    MEAN_STDS = \
        {
            'CIR':
            (
                (0.2927, 0.3166, 0.3368),
                (0.1507, 0.1799, 0.1966)
            ),
            'R': (0.3368, 0.1966),
            'G': (0.3166, 0.1799),
            'NIR': (0.2927, 0.1507),
            'RE': (0.3152, 0.1563)
        }

    def __init__(self, dataset_params, name="sequoia"):
        super(SequoiaDatasetInterface, self).__init__(dataset_params)
        channels = dataset_params['channels']
        self.dataset_name = name

        mean, std = self.MEAN_STDS[channels] if type(channels) is str else \
            list(zip(*(self.MEAN_STDS[c] for c in channels)))

        self.lib_dataset_params = {
            'mean': mean,
            'std': std
        }

        # crop_size = core_utils.get_param(self.dataset_params, 'crop_size', default_val=320)

        transform_train = transforms.Compose([
            PairRandomFlip(orientation="horizontal"),
            # PairRandomFlip(orientation="vertical"),
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std']),
        ])

        target_transform = transforms.Compose([
            PairRandomFlip(orientation="horizontal"),
            # PairRandomFlip(orientation="vertical"),
            transforms.PILToTensor(),
            squeeze0,
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

        self.trainset = SequoiaDataset(root=self.dataset_params.root, train=True, channels=channels,
                                       batch_size=self.dataset_params.batch_size, index=train_index,
                                       transform=transform_train, target_transform=target_transform)

        self.valset = SequoiaDataset(root=self.dataset_params.root, train=False, channels=channels,
                                     batch_size=self.dataset_params.val_batch_size, index=val_index,
                                     transform=transform_test, target_transform=target_transform)

        self.testset = SequoiaDataset(root=self.dataset_params.root, train=False, channels=channels,
                                      batch_size=self.dataset_params.test_batch_size, index=test_index,
                                      transform=transform_test, target_transform=target_transform)

    def undo_preprocess(self, x):
        return (Denormalize(self.lib_dataset_params['mean'], self.lib_dataset_params['std'])(x) * 255).type(torch.uint8)

    def build_data_loaders(self, batch_size_factor=1, num_workers=8, train_batch_size=None, val_batch_size=None,
                           test_batch_size=None, distributed_sampler: bool = False):
        """

        define train, val (and optionally test) loaders. The method deals separately with distributed training and standard
        (non distributed, or parallel training). In the case of distributed training we need to rely on distributed
        samplers.
        :param batch_size_factor: int - factor to multiply the batch size (usually for multi gpu)
        :param num_workers: int - number of workers (parallel processes) for dataloaders
        :param train_batch_size: int - batch size for train loader, if None will be taken from dataset_params
        :param val_batch_size: int - batch size for val loader, if None will be taken from dataset_params
        :param distributed_sampler: boolean flag for distributed training mode
        :return: train_loader, val_loader, classes: list of classes
        """
        # CHANGE THE BATCH SIZE ACCORDING TO THE NUMBER OF DEVICES - ONLY IN NON-DISTRIBUED TRAINING MODE
        # IN DISTRIBUTED MODE WE NEED DISTRIBUTED SAMPLERS
        # NO SHUFFLE IN DISTRIBUTED TRAINING
        if distributed_sampler:
            self.batch_size_factor = 1
            train_sampler = DistributedSampler(self.trainset)
            val_sampler = DistributedSampler(self.valset)
            test_sampler = DistributedSampler(self.testset) if self.testset is not None else None
            train_shuffle = False
        else:
            self.batch_size_factor = batch_size_factor
            train_sampler = None
            val_sampler = None
            test_sampler = None
            train_shuffle = True

        if train_batch_size is None:
            train_batch_size = self.dataset_params.batch_size * self.batch_size_factor
        if val_batch_size is None:
            val_batch_size = self.dataset_params.val_batch_size * self.batch_size_factor
        if test_batch_size is None:
            test_batch_size = self.dataset_params.test_batch_size * self.batch_size_factor

        train_loader_drop_last = core_utils.get_param(self.dataset_params, 'train_loader_drop_last', default_val=False)

        cutmix = core_utils.get_param(self.dataset_params, 'cutmix', False)
        cutmix_params = core_utils.get_param(self.dataset_params, 'cutmix_params')

        # WRAPPING collate_fn
        train_collate_fn = core_utils.get_param(self.trainset, 'collate_fn')
        val_collate_fn = core_utils.get_param(self.valset, 'collate_fn')
        test_collate_fn = core_utils.get_param(self.testset, 'collate_fn')

        if cutmix and train_collate_fn is not None:
            raise IllegalDatasetParameterException("cutmix and collate function cannot be used together")

        if cutmix:
            # FIXME - cutmix should be available only in classification dataset. once we make sure all classification
            # datasets inherit from the same super class, we should move cutmix code to that class
            logger.warning("Cutmix/mixup was enabled. This feature is currently supported only "
                           "for classification datasets.")
            train_collate_fn = CollateMixup(**cutmix_params)

        # FIXME - UNDERSTAND IF THE num_replicas VARIBALE IS NEEDED
        # train_sampler = DistributedSampler(self.trainset,
        #                                    num_replicas=distributed_gpus_num) if distributed_sampler else None
        # val_sampler = DistributedSampler(self.valset,
        #                                   num_replicas=distributed_gpus_num) if distributed_sampler else None
        pw = num_workers > 0
        self.train_loader = torch.utils.data.DataLoader(self.trainset,
                                                        batch_size=train_batch_size,
                                                        shuffle=train_shuffle,
                                                        num_workers=num_workers,
                                                        pin_memory=True,
                                                        sampler=train_sampler,
                                                        collate_fn=train_collate_fn,
                                                        drop_last=train_loader_drop_last,
                                                        persistent_workers=pw)

        self.val_loader = torch.utils.data.DataLoader(self.valset,
                                                      batch_size=val_batch_size,
                                                      shuffle=False,
                                                      num_workers=num_workers,
                                                      pin_memory=True,
                                                      sampler=val_sampler,
                                                      collate_fn=val_collate_fn,
                                                      persistent_workers=pw)

        if self.testset is not None:
            self.test_loader = torch.utils.data.DataLoader(self.testset,
                                                           batch_size=test_batch_size,
                                                           shuffle=False,
                                                           num_workers=num_workers,
                                                           pin_memory=True,
                                                           sampler=test_sampler,
                                                           collate_fn=test_collate_fn,
                                                           persistent_workers=pw)

        self.classes = self.trainset.classes


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
            self.index = self.build_index(root, channels=channels)

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
        return Image.fromarray(
            np.moveaxis(
                np.stack([
                    Image.open(
                        os.path.join(self.path, folder, 'tile', c, file)
                    ) for c in self.channels
                ]
                ),
                0, -1)
        )

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
