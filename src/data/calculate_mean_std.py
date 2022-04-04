import multiprocessing

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms

from data.sequoia import SequoiaDataset

WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 256
WIDTH = 360
HEIGHT = 480


def calculate():
    """
    Calculate the mean and the standard deviation of a dataset
    """
    augs = transforms.Compose([
                      transforms.ToTensor()])

    sq = SequoiaDataset("./dataset/processed/Sequoia", transform=augs, target_transform=augs)
    count = len(sq) * WIDTH * HEIGHT

    try:
        # data loader
        image_loader = DataLoader(sq,
                                  batch_size=BATCH_SIZE,
                                  shuffle=False,
                                  num_workers=WORKERS,
                                  pin_memory=True)

        # placeholders
        psum = torch.tensor([0.0, 0.0, 0.0])
        psum_sq = torch.tensor([0.0, 0.0, 0.0])

        # loop through images
        for input, _ in tqdm(image_loader):
            psum += input.sum(axis=[0, 2, 3])
            psum_sq += (input ** 2).sum(axis=[0, 2, 3])

    finally:
        total_mean = psum / count
        total_var = (psum_sq / count) - (total_mean ** 2)
        total_std = torch.sqrt(total_var)

        print("Mean: {}".format(total_mean))
        print("Var : {}".format(total_var))
        print("Std : {}".format(total_std))


if __name__ == '__main__':
    calculate()
