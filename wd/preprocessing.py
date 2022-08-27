import numpy as np
import torch
from PIL import Image

from wd.data.sequoia import WeedMapDataset
from tqdm import tqdm

import torchvision.transforms as transforms
import os
import shutil


def delete_empty_imgs(root, channels, tempdir_check=None):
    if tempdir_check:
        shutil.rmtree(tempdir_check, ignore_errors=True)
        os.makedirs(tempdir_check, exist_ok=True)
    trs = transforms.Compose([])
    dataset = WeedMapDataset(root, transform=trs, return_mask=True, target_transform=trs)
    counter = 0
    for i in tqdm(range(len(dataset))):
        folder, img_name = dataset.index[i]
        img, (gt, mask) = dataset[i]
        if np.min(np.array(mask)) == 255:
            gt.close()
            mask.close()
            gt_path_color = os.path.join(root, folder, 'groundtruth',
                                         folder + '_' + img_name.split('.')[0] + '_GroundTruth_color.png'
                                         )
            gt_path_imap = os.path.join(root, folder, 'groundtruth',
                                        folder + '_' + img_name.split('.')[0] + '_GroundTruth_iMap.png'
                                        )
            gt_path = os.path.join(root, folder, 'groundtruth',
                                   folder + '_' + img_name)
            mask_path = os.path.join(root, folder, 'mask', img_name)

            if tempdir_check:
                if isinstance(img, torch.Tensor):
                    img = Image.fromarray(img.byte().permute(1, 2, 0).numpy())
                img.save(os.path.join(tempdir_check, img_name))
                shutil.copy(gt_path_color, os.path.join(tempdir_check,
                                                        folder + '_' + img_name.split('.')[0] + '_GroundTruth_color.png'))
            os.remove(gt_path_color)
            os.remove(gt_path_imap)
            os.remove(gt_path)
            os.remove(mask_path)
            for c in channels:
                c_path = os.path.join(root, folder, 'tile', c, img_name)
                os.remove(c_path)
            counter += 1
    print('Removed {} empty images'.format(counter))


def copy_dataset(inpath, outpath):
    shutil.copytree(inpath, outpath, dirs_exist_ok=True)
    print('Dataset copied')


def preprocess(subset):
    if subset == "SEQUOIA":
        # SEQUOIA
        channels = ['CIR', 'G', 'NDVI', 'NIR', 'R', 'RE']
        path = 'dataset/raw/Sequoia'
        outpath = 'dataset/processed/Sequoia'
    elif subset == "REDEDGE":
        # REDEDGE 413 removed
        channels = ['CIR', 'G', 'NDVI', 'NIR', 'R', 'RE', 'B']
        path = 'dataset/raw/RedEdge'
        outpath = 'dataset/processed/RedEdge'
        copy_dataset(path, outpath)
    else:
        raise NotImplementedError()
    delete_empty_imgs(outpath, channels, tempdir_check='tmp')


