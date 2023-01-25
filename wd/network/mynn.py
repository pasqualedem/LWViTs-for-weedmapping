"""
Custom Norm wrappers to enable sync BN, regular BN and for weight
initialization
"""
import torch
import torch.nn as nn

from wd.network.config import cfg
import torch.amp


align_corners = cfg.MODEL.ALIGN_CORNERS


def Norm2d(in_channels, **kwargs):
    """
    Custom Norm Function to allow flexible switching
    """
    return torch.nn.BatchNorm2d(in_channels, **kwargs)


def initialize_weights(*models):
    """
    Initialize Model Weights
    """
    for model in models:
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, cfg.MODEL.BNFUNC):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=align_corners).float()


def Upsample2(x):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                                     align_corners=align_corners).float()


def Down2x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=0.5, mode='bilinear', align_corners=align_corners)


def Up15x(x):
    return torch.nn.functional.interpolate(
        x, scale_factor=1.5, mode='bilinear', align_corners=align_corners)


def scale_as(x, y):
    '''
    scale x to the same size as y
    '''
    y_size = y.size(2), y.size(3)

    return torch.nn.functional.interpolate(
        x, size=y_size, mode='bilinear', align_corners=align_corners
    )


def DownX(x, scale_factor):
    '''
    scale x to the same size as y
    '''
    return (
        torch.nn.functional.interpolate(
            x,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=align_corners,
            recompute_scale_factor=True,
        )
    )


def ResizeX(x, scale_factor):
    '''
    scale x by some factor
    '''
    return (
        torch.nn.functional.interpolate(
            x,
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=align_corners,
            recompute_scale_factor=True,
        )
    )
