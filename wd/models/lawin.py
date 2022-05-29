import torch
from super_gradients.training.utils import get_param
from torch import Tensor
from torch.nn import functional as F
from wd.models.base import BaseModel
from wd.models.heads.lawin import LawinHead
from wd.models.heads.laweed import LaweedHead


class BaseLawin(BaseModel):
    """
    Abstract base lawin class with free decoder head lawin based
    """
    def __init__(self, arch_params, lawin_class) -> None:
        num_classes = get_param(arch_params, "num_classes")
        input_channels = get_param(arch_params, "input_channels", 3)
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        backbone_pretrained = get_param(arch_params, "backbone_pretrained", False)
        super().__init__(backbone, input_channels, backbone_pretrained)
        self.decode_head = lawin_class(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.apply(self._init_weights)
        if backbone_pretrained:
            self.backbone.init_pretrained_weights()

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y


class Lawin(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class Laweed(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


if __name__ == '__main__':
    model = Lawin('MiT-B1')
    model.eval()
    x = torch.zeros(1, 3, 512, 512)
    y = model(x)
    print(y.shape)
    from fvcore.nn import flop_count_table, FlopCountAnalysis
    print(flop_count_table(FlopCountAnalysis(model, x)))