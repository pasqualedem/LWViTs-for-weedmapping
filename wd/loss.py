from torch.nn import CrossEntropyLoss, Module
import torch.nn.functional as F
import torch


class CELoss(CrossEntropyLoss):
    def __init__(self, *args, weight=None, **kwargs):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(*args, weight=weight, **kwargs)

    def __call__(self, x, target, **kwargs):
        return super().__call__(x, target.float(), **kwargs)


class FocalLoss(Module):
    def __init__(self, alpha: float = 1, gamma: float = 0, weight=None, reduction: str = 'none', **kwargs):
        super().__init__()
        if weight:
            self.weight = torch.tensor(weight)
        self.gamma = gamma

        if reduction == 'none':
            self.reduction = lambda x: x
        elif reduction == 'mean':
            self.reduction = torch.mean
        elif reduction == 'sum':
            self.reduction = torch.sum
        else:
            raise NotImplementedError(f"Invalid reduction mode: {self.reduction}")

    def __call__(self, x, target, **kwargs):
        soft = F.softmax(x, 1)
        if self.weight is not None:
            wtarget = self.weight * target
        else:
            wtarget = target

        pt = soft[wtarget != 0]
        ce = -(torch.log(pt) * wtarget[wtarget != 0])

        focus = torch.pow(-soft + 1.0, self.gamma)
        focal = (focus * ce)

        return self.reduction(focal)


LOSSES = {
    'cross_entropy': CELoss,
    'focal': FocalLoss
}
