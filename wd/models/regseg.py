import torch
import torch.nn as nn
from super_gradients.training.models import SgModule
from super_gradients.training.models.segmentation_models.regseg import RegSeg, DEFAULT_REGSEG48_BACKBONE_PARAMS, \
    DEFAULT_REGSEG48_DECODER_PARAMS, RegSegBackbone, RegSegDecoder, RegSegHead, DEFAULT_REGSEG48_HEAD_PARAMS
from super_gradients.training.utils import HpmStruct, get_param
from super_gradients.training.utils.module_utils import ConvBNReLU


class RegSeg48(RegSeg):

    def __init__(self, arch_params: HpmStruct):
        num_classes = get_param(arch_params, "num_classes")
        input_channels = get_param(arch_params, "input_channels", 3)
        stem = ConvBNReLU(in_channels=input_channels, out_channels=32, kernel_size=3, stride=2, padding=1)
        backbone = RegSegBackbone(in_channels=32, backbone_config=DEFAULT_REGSEG48_BACKBONE_PARAMS)
        decoder = RegSegDecoder(
            backbone.get_backbone_output_number_of_channels(),
            DEFAULT_REGSEG48_DECODER_PARAMS
        )
        head = RegSegHead(decoder.out_channels, num_classes, DEFAULT_REGSEG48_HEAD_PARAMS)
        super().__init__(stem, backbone, decoder, head)

    def replace_head(self, new_num_classes: int, head_config: dict = None):
        head_config = head_config or DEFAULT_REGSEG48_HEAD_PARAMS
        super().replace_head(new_num_classes, head_config)
