import torch
from super_gradients.training.utils import get_param, HpmStruct
from super_gradients.training import utils as sg_utils
from torch import Tensor
from torch.nn import functional as F

from ezdl.utils.utilities import filter_none
from ezdl.models.backbones.mit import MiTFusion
from ezdl.models.base import BaseModel
from ezdl.models.heads.lawin import LawinHead
from ezdl.models.heads.laweed import LaweedHead


class BaseLawin(BaseModel):
    """
    Abstract base lawin class with free decoder head lawin based
    """

    def __init__(self, arch_params, lawin_class) -> None:
        num_classes = get_param(arch_params, "num_classes")
        input_channels = get_param(arch_params, "input_channels", 3)
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        backbone_pretrained = get_param(arch_params, "backbone_pretrained", False)
        pretrained_channels = get_param(arch_params, "main_pretrained", None)
        super().__init__(backbone, input_channels, backbone_pretrained)
        self.decode_head = lawin_class(self.backbone.channels, 256 if 'B0' in backbone else 512, num_classes)
        self.apply(self._init_weights)
        if backbone_pretrained:
            self.main_pretrained = pretrained_channels
            if isinstance(pretrained_channels, str):
                self.main_pretrained = [pretrained_channels] * input_channels
            else:
                self.main_pretrained = pretrained_channels
            self.backbone.init_pretrained_weights(self.main_pretrained)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
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


class BaseDoubleLawin(BaseLawin):
    """
    Notes::::: This implementation has larger params and FLOPs than the results reported in the paper.
    Will update the code and weights if the original author releases the full code.
    """

    def __init__(self, arch_params, lawin_class) -> None:
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        main_channels = get_param(arch_params, "main_channels", None)
        if main_channels is None:
            raise ValueError("Please provide main_channels")
        self.side_channels = arch_params['input_channels'] - main_channels
        self.side_pretrained = get_param(arch_params, "side_pretrained", None)
        self.main_channels = main_channels
        arch_params['input_channels'] = arch_params['main_channels']
        super().__init__(arch_params, lawin_class)
        self.side_backbone = self.eval_backbone(backbone, self.side_channels, pretrained=bool(self.side_pretrained))
        if self.side_pretrained is not None:
            if isinstance(self.side_pretrained, str):
                self.side_pretrained = [self.side_pretrained] * self.side_channels
            self.side_backbone.init_pretrained_weights(self.side_pretrained)
        p_local = get_param(arch_params, "p_local", None)
        p_glob = get_param(arch_params, "p_glob", None)
        fusion_type = get_param(arch_params, "fusion_type", None)
        self.fusion = MiTFusion(self.backbone.channels,
                                **filter_none({"p_local": p_local, "p_glob": p_glob, "fusion_type": fusion_type}))

    def forward(self, x: Tensor) -> Tensor:
        main_channels = x[:, :self.main_channels, ::].contiguous()
        side_channels = x[:, self.main_channels:, ::].contiguous()
        feat_main = self.backbone(main_channels)
        feat_side = self.side_backbone(side_channels)
        feat = self.fusion((feat_main, feat_side))
        y = self.decode_head(feat)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y


class DoubleLawin(BaseDoubleLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class DoubleLaweed(BaseDoubleLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


class BaseSplitLawin(BaseLawin):
    def __init__(self, arch_params, lawin_class) -> None:
        backbone = get_param(arch_params, "backbone", 'MiT-B0')
        main_channels = get_param(arch_params, "main_channels", None)
        if main_channels is None:
            raise ValueError("Please provide main_channels")
        self.side_channels = arch_params['input_channels'] - main_channels
        self.side_pretrained = get_param(arch_params, "side_pretrained", None)
        self.main_channels = main_channels
        arch_params['input_channels'] = arch_params['main_channels']
        super().__init__(arch_params, lawin_class)
        self.side_backbone = self.eval_backbone(backbone, self.side_channels,
                                                n_blocks=1,
                                                pretrained=bool(self.side_pretrained))
        if self.side_pretrained is not None:
            if isinstance(self.side_pretrained, str):
                self.side_pretrained = [self.side_pretrained] * self.side_channels
            self.side_backbone.init_pretrained_weights(self.side_pretrained)
        p_local = get_param(arch_params, "p_local", None)
        p_glob = get_param(arch_params, "p_glob", None)
        fusion_type = get_param(arch_params, "fusion_type", None)
        self.fusion = MiTFusion(self.backbone.channels,
                                **filter_none({"p_local": p_local, "p_glob": p_glob, "fusion_type": fusion_type}))

    def forward(self, x: Tensor) -> Tensor:
        main_channels = x[:, :self.main_channels, ::].contiguous()
        side_channels = x[:, self.main_channels:, ::].contiguous()
        first_feat_side = self.side_backbone(side_channels)
        first_feat_main = self.backbone.partial_forward(main_channels, slice(0, 1))
        first_feat = self.fusion((first_feat_main, first_feat_side))[0]
        feat = (first_feat,) + self.backbone.partial_forward(first_feat, slice(1, 4))
        y = self.decode_head(feat)  # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)  # to original image shape
        return y

    def initialize_param_groups(self, lr: float, training_params: HpmStruct) -> list:
        """

        :return: list of dictionaries containing the key 'named_params' with a list of named params
        """

        def f(x):
            return not (x[0].startswith('backbone') and int(x[0].split('.')[4]) == 0)

        freeze_pretrained = sg_utils.get_param(training_params, 'freeze_pretrained', False)
        if self.backbone_pretrained and freeze_pretrained:
            return [{'named_params': list(filter(f, list(self.named_parameters())))}]
        return [{'named_params': self.named_parameters()}]


class SplitLawin(BaseSplitLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LawinHead)


class SplitLaweed(BaseSplitLawin):
    def __init__(self, arch_params) -> None:
        super().__init__(arch_params, LaweedHead)


# Legacy names
lawin = Lawin
laweed = Laweed
doublelawin = DoubleLawin
doublelaweed = DoubleLaweed
splitlawin = SplitLawin
splitlaweed = SplitLaweed
