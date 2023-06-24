from ezdl.models import Lawin
from ezdl.datasets import WeedMapDatasetInterface
from ezdl.metrics import F1, Precision, Recall
import torch

from tqdm import tqdm


def crosstest():
    # 1. Create a model
    device = "cuda"
    model = Lawin({
        "backbone": "MiT-B1", 
        "backbone_pretrained": True,
        "input_channels": 1,
        "pretrained": {
            "source": "file",
            "file": "checkpoints/Sequoia_B1_NDVI.pth",
        },
        "num_classes": 3,
        }).eval().to(device)
    # 2. Create a dataset interface
    dataset = WeedMapDatasetInterface(dict(
        root="../../PHD/Datasets/WeedMap/0_rotations_processed_003_test/RedEdge",
        channels=["NDVI"],
        train_folders=["000", "001", "002", "004"],
        test_folders=["003"],
        batch_size=1,
        hor_flip=False,
        ver_flip=False,
        return_path=True,
        ))
    dataset.build_data_loaders(num_workers=0)
    metric_args = dict(num_classes=3)
    f1, precision, recall = F1(**metric_args), Precision(**metric_args), Recall(**metric_args)
    with torch.no_grad():
        for batch in tqdm(dataset.test_loader):
            image, mask, additional = batch
            image = image.to(device)
            mask = mask.to(device)
            prediction = model(image)
            f1.update(prediction, mask)
            precision.update(prediction, mask)
            recall.update(prediction, mask)
    print(f"F1: {f1.compute()}")
    print(f"Precision: {precision.compute()}")
    print(f"Recall: {recall.compute()}")

