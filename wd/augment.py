import multiprocessing
import os

import PIL.Image
import torch

from tqdm import tqdm
from torchvision.transforms import transforms
from torchvision.transforms import functional as F, InterpolationMode

from wd.data.weedmap import WeedMapDataset

WORKERS = multiprocessing.cpu_count()
WEED_CLASS = 2
ROTATIONS_REDEDGE = 0
ROTATIONS_SEQUOIA = 4
CROP_SIZE = [256, 256]
CIR = [3, 1, 0]


def get_crops(img, size):
    if isinstance(img, PIL.Image.Image):
        h, w = img.size
    elif isinstance(img, torch.Tensor):
        w, h = img.size()[-2:]
    else:
        raise TypeError("img should be PIL.Image or torch.Tensor, not {}".format(type(img)))
    yield 0, F.crop(img, 0, 0, *size)
    yield 1, F.crop(img, 0, h - size[1], *size)
    yield 2, F.crop(img, w - size[0], 0, *size)
    yield 3, F.crop(img, w - size[0], h - size[1], *size)


def generate_train(root, target_root, train_folders, channels):
    index = WeedMapDataset.build_index(
        root,
        macro_folders=train_folders,
        channels=channels,
    )

    sq = WeedMapDataset(root,
                        transform=lambda x: x,
                        target_transform=lambda x: x,
                        index=index,
                        channels=channels,
                        return_path=True
                        )

    init_degrees = [-180, 180]

    totensor = transforms.ToTensor()
    # loop through images
    images_with_weed = 0
    for input, target, additional in tqdm(sq):
        for (k, input_crop), (_, target_crop) in zip(get_crops(input, CROP_SIZE), get_crops(target, CROP_SIZE)):
            f, name = additional['input_name'].split('_')
            name = name.split('.')[0]

            if WEED_CLASS in totensor(target_crop).unique():
                for r in range(ROTATIONS):
                    degrees = torch.FloatTensor(1).uniform_(*init_degrees).item()
                    r_input = F.rotate(input_crop, degrees, InterpolationMode.BILINEAR)
                    r_target = F.rotate(target_crop, degrees, InterpolationMode.NEAREST)
                    for i, c in enumerate(r_input):
                        F.to_pil_image(c).save(os.path.join(target_root, f, "tile", channels[i], f"{name}_{k}_{r}.png"))
                    r_target.save(os.path.join(target_root, f, "groundtruth", f"{f}_{name}_{k}_{r}_GroundTruth_iMap.png"))
                    F.to_pil_image(torch.stack([r_input[CIR[0]], r_input[CIR[1]], r_input[CIR[2]]]))\
                        .save(os.path.join(target_root, f, "tile", 'CIR', f"{name}_{k}_{r}.png"))
                images_with_weed += 1

            for i, c in enumerate(input_crop):
                F.to_pil_image(c).save(os.path.join(target_root, f, "tile", channels[i], f"{name}_{k}.png"))
            target_crop.save(os.path.join(target_root, f, "groundtruth", f"{f}_{name}_{k}_GroundTruth_iMap.png"))
            F.to_pil_image(torch.stack([input_crop[CIR[0]], input_crop[CIR[1]], input_crop[CIR[2]]])) \
                .save(os.path.join(target_root, f, "tile", 'CIR', f"{name}_{k}.png"))

    print(f"{images_with_weed} images with weed on {len(sq)} images")  # 363 images with weed


def generate_test(root, target_root, test_folders, channels):
    index = WeedMapDataset.build_index(
        root,
        macro_folders=test_folders,
        channels=channels,
    )

    sq = WeedMapDataset(root,
                        transform=lambda x: x,
                        target_transform=lambda x: x,
                        index=index,
                        channels=channels,
                        return_path=True
                        )
    # loop through images
    for input, target, additional in tqdm(sq):
        for (k, input_crop), (_, target_crop) in zip(get_crops(input, CROP_SIZE), get_crops(target, CROP_SIZE)):
            f, name = additional['input_name'].split('_')
            name = name.split('.')[0]

            for i, c in enumerate(input_crop):
                F.to_pil_image(c).save(os.path.join(target_root, f, "tile", channels[i], f"{name}_{k}.png"))
            target_crop.save(os.path.join(target_root, f, "groundtruth", f"{f}_{name}_{k}_GroundTruth_iMap.png"))
            F.to_pil_image(torch.stack([input_crop[CIR[0]], input_crop[CIR[1]], input_crop[CIR[2]]])) \
                .save(os.path.join(target_root, f, "tile", 'CIR', f"{name}_{k}.png"))


def augment(subset):
    """
    Calculate the mean and the standard deviation of a dataset
    """
    torch.manual_seed(42)
    global ROTATIONS

    # Sequoia 277 images with weed on 337 (x4) images
    if subset == "Sequoia":
        ROTATIONS = ROTATIONS_SEQUOIA
        root = "./dataset/processed/Sequoia"
        train_folders = ['006', '007']
        test_folders = ['005']
        target_root = f"./dataset/{ROTATIONS}_rotations_processed_{'-'.join(test_folders)}_test/Sequoia"
        channels = ['R', 'G', 'NDVI', 'NIR', 'RE']
    elif subset == "RedEdge":
        # RedEdge 1285 images with weed on 455 (x4) images
        ROTATIONS = ROTATIONS_REDEDGE
        root = "./dataset/processed/RedEdge"
        train_folders = ['000', '001', '002', '004']
        test_folders = ['003']
        target_root = f"./dataset/{ROTATIONS}_rotations_processed_{'-'.join(test_folders)}_test/RedEdge"
        channels = ['R', 'G', 'B', 'NDVI', 'NIR', 'RE']
    else:
        raise Exception("Subset not recognized")

    os.makedirs(target_root, exist_ok=True)
    for f in train_folders + test_folders:
        os.makedirs(os.path.join(target_root, f), exist_ok=True)
        for c in channels:
            os.makedirs(os.path.join(target_root, f, "tile", c), exist_ok=True)
        os.makedirs(os.path.join(target_root, f, "groundtruth"), exist_ok=True)
        os.makedirs(os.path.join(target_root, f, "tile", "CIR"), exist_ok=True)

    generate_train(root, target_root, train_folders, channels)
    generate_test(root, target_root, test_folders, channels)

