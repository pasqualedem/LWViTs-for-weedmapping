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
    def __init__(self, gamma: float = 0, weight=None, reduction: str = 'mean', **kwargs):
        super().__init__()
        self.weight = None
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
        ce_loss = F.cross_entropy(x, target.float(), reduction='none', **kwargs)
        pt = torch.exp(-ce_loss)
        if self.weight is not None:
            wtarget = self.weight[(...,) + (None, ) * (len(target.shape) - 1)]\
                          .moveaxis(0, 1).to(target.device) \
                      * target
            wtarget = wtarget[wtarget != 0].reshape(pt.shape)
            focal_loss = torch.pow((1 - pt), self.gamma) * wtarget * ce_loss
        else:
            focal_loss = torch.pow((1 - pt), self.gamma) * ce_loss

        return self.reduction(focal_loss)


LOSSES = {
    'cross_entropy': CELoss,
    'focal': FocalLoss
} 
