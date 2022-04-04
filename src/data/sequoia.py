import os

import numpy as np
import tensorflow as tf
from PIL import Image

PATH = "/dataset/Sequoia"


class SequoiaDataset(tf.keras.utils.Sequence):
    def __init__(
            self,
            path,
            channels=('R', 'G'),
            batch_size=512,
            shuffle=False,
            seed=42
    ):
        """
        Initialize a sequence of Graph User-Item IDs.

        :param batch_size: The batch size.
        :param shuffle: Whether to shuffle the sequence.
        :param seed: The seed value used to shuffle the sequence.
        """
        super().__init__()
        macro_folders = os.listdir(path)
        self.len = len(os.listdir(os.path.join(path, macro_folders[0], 'tile', channels[0])))
        self.channels = channels
        self.n_files = {}
        self.path = path
        for folder in macro_folders:
            self.n_files[folder] = (os.listdir(os.path.join(self.path, folder, 'tile', self.channels[0])))
        self.index = dict()
        i = 0
        for folder, files in self.n_files.items():
            for file in files:
                self.index[i] = folder, file
                i += 1

        # Set other settings
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.indexes = None
        self.random_state = None
        self.on_epoch_end()

    def __len__(self):
        """
        Get the number of batches.

        :return: The number of batches.
        """
        return int(np.ceil(self.len / self.batch_size))

    def __getitem__(self, idx):
        """
        Get the i-th batch consisting of User-Item IDs and the ratings.

        :param idx: The index of the batch.
        :return: A pair consisting of User-Item IDs and the ratings.
        """
        batch_idx = idx * self.batch_size
        batch_off = min(batch_idx + self.batch_size, len(self.index))
        imgs = []
        gts = []
        masks = []
        for i in range(batch_idx, batch_off):
            for c in self.channels:
                folder, file = self.index[i]
                img = Image.open(
                    os.path.join(self.path, folder, 'tile', c, file)
                )
                mask = Image.open(
                    os.path.join(self.path, folder, 'mask', file)
                )
                gt = Image.open(
                    os.path.join(self.path, folder, 'groundtruth',
                                 folder + '_' + file.split('.')[0] + '_GroundTruth_color.png'
                                 )
                )
                imgs.append(np.array(img))
                masks.append(mask)
                gts.append(np.array(gt))
        # return imgs, masks, gts
        return np.array(imgs), np.array(gts)

    def on_epoch_end(self):
        """
        Shuffles the indexes at the end of every epoch.
        """
        if self.shuffle:
            if self.random_state is None:
                self.random_state = np.random.RandomState(self.seed)
            self.indexes = np.arange(len(self.ratings))
            self.random_state.shuffle(self.indexes)
