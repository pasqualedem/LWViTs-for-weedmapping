import torch
import math
from torch import nn
from wd.models.backbones import *
from wd.models.layers import trunc_normal_


class BaseModel(nn.Module):
    def __init__(self, backbone: str = 'MiT-B0', input_channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        backbone, variant = backbone.split('-')
        self.backbone = eval(backbone)(variant, input_channels, pretrained)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out // m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.backbone.load_state_dict(torch.load(pretrained, map_location='cpu'), strict=False)