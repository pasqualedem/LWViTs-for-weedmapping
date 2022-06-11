import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import functional as F

class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0, d=1, g=1):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, d, g, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Copied from timm
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    def __init__(self, p: float = None):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0. or not self.training:
            return x
        kp = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        return x.div(kp) * random_tensor


class MultiDropPath(nn.Module):
    def __init__(self, num_inputs, p: float = None):
        super().__init__()
        self.p = p
        self.num_inputs = num_inputs
        self.c = torch.distributions.Categorical(torch.tensor([1/num_inputs] * num_inputs))

    def forward(self, inputs: Tensor) -> list:
        if self.p == 0. or not self.training:
            return inputs
        kp = 1 - self.p
        x = inputs[0]
        shape = (len(inputs), x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = kp + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize

        choice_mask = random_tensor.any(dim=0) \
            .repeat(((self.num_inputs,) + (1,) * (x.ndim - 1))) \
            .logical_not()
        choice_mask = rearrange(choice_mask, "(i b) ... -> i b ...", i=len(inputs))

        choice = F.one_hot(self.c.sample([x.shape[0]]).to(x.device), num_classes=self.num_inputs)\
            .T.reshape(choice_mask.shape)

        mask = choice_mask.logical_and(choice)
        random_tensor = mask.logical_or(random_tensor)
        return [inputs[i].div(kp + ((1-kp)**self.num_inputs)/self.num_inputs) * random_tensor[i]
                for i in range(len(inputs))]
