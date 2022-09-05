import os.path

import torch
from PIL import Image
from super_gradients.training import StrictLoad
from super_gradients.training.utils.checkpoint_utils import load_checkpoint_to_model
from super_gradients.training import utils as core_utils
from torchvision.transforms import functional as F
import torchvision.transforms as transforms

from wd.data.sequoia import WeedMapDatasetInterface
from wd.utils.utilities import load_yaml

from wd.models import MODELS as MODELS_DICT


class Inferencer:
    def __init__(self, name, path, params, gpu=True):
        self.preprocess = None

        if name in MODELS_DICT.keys():
            self.model = MODELS_DICT[name](params)
        else:
            self.model = name
        self.model = core_utils.WrappedModel(self.model)
        _ = load_checkpoint_to_model(ckpt_local_path=path,
                                     load_backbone=False,
                                     net=self.model,
                                     strict=StrictLoad.ON.value,
                                     load_weights_only=True)
        self.model.eval()
        if gpu:
            self.model.cuda()

    def set_preprocess(self, preprocess_fn):
        self.preprocess = preprocess_fn

    def __call__(self, image, preprocess=True):
        if preprocess:
            image = self.preprocess(image)
        return self.model(image)


class WandbInferencer(Inferencer):
    config_file = 'config.yaml'
    config_key = 'value'
    wb_input_params = 'in_params'

    def __init__(self, path_wrapper, config_wrapper, gpu=True):
        self.channels = None
        config = load_yaml(config_wrapper.name)[self.wb_input_params]
        if self.config_key in config:
            config = config[self.config_key]

        input_channels = len(config['dataset']['channels'])
        output_channels = config['dataset']['num_classes']
        params = {
            'input_channels': input_channels,
            'output_channels': output_channels,
            'in_channels': input_channels,
            'out_channels': output_channels,
            'num_classes': output_channels,
            **config['model']['params']
        }
        super().__init__(name=config['model']['name'], path=path_wrapper.name, params=params, gpu=gpu)
        self.compose_preprocess(config)

    def compose_preprocess(self, config):
        means, stds = WeedMapDatasetInterface.get_mean_std(
            config["dataset"]["train_folders"],
            config["dataset"]["channels"],
            os.path.basename(config["dataset"]["root"].lower()))
        self.channels = config["dataset"]["channels"]
        self.preprocess = transforms.Normalize(means, stds)


def compose_images(images: list):
    if len(images) == 1:
        return F.to_tensor(images[0]).unsqueeze(0)
    else:
        return torch.stack([F.to_tensor(img).squeeze(0) for img in images]).unsqueeze(0)
