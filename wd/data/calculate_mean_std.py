import multiprocessing

import torch
import json

from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import transforms

from wd.data.sequoia import WeedMapDataset

WORKERS = multiprocessing.cpu_count()
BATCH_SIZE = 256
WIDTH = 360
HEIGHT = 480


def calculate(root, folders, channels):
    """
    Calculate the mean and the standard deviation of a dataset
    """

    augs = transforms.Compose([
                      transforms.ToTensor()])

    index = WeedMapDataset.build_index(
        root,
        macro_folders=folders,
        channels=channels,
    )

    sq = WeedMapDataset(root,
                        transform=lambda x: x,
                        target_transform=augs,
                        index=index,
                        channels=channels
                        )
    count = len(sq) * WIDTH * HEIGHT


    # data loader
    image_loader = DataLoader(sq,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=True)

    # placeholders
    psum = torch.zeros(len(channels))
    psum_sq = torch.zeros(len(channels))

    # loop through images
    for input, _ in tqdm(image_loader):
        psum += input.sum(axis=[0, 2, 3])
        psum_sq += (input ** 2).sum(axis=[0, 2, 3])

    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    d = {}
    for i in range(len(channels)):
        d[channels[i]] = {
            'mean': total_mean[i].item(),
            'std': total_std[i].item(),
            'sum': psum[i].item(),
            'sum_sq': psum_sq[i].item(),
        }
    d['count'] = count
    return d


if __name__ == '__main__':
    # SEQUOIA
    # root = "./dataset/processed/Sequoia"
    # folders = ['005', '006', '007']
    # channels = ['R', 'G', 'NDVI', 'NIR', 'RE']

    # REDEDGE
    root = "./dataset/processed/RedEdge"
    folders = ['000', '001', '002', '004']
    channels = ['R', 'G', 'B', 'NDVI', 'NIR', 'RE']
    d = {}
    for folder in folders:
        d[folder] = calculate(root, [folder], channels)
    print(json.dumps(d, indent=4))