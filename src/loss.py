from torch.nn import CrossEntropyLoss
import torch


class CELoss(CrossEntropyLoss):
    def __init__(self, *args, weight=None, **kwargs):
        if weight:
            weight = torch.tensor(weight)
        super().__init__(*args, weight=weight, **kwargs)

    def __call__(self, x, target, **kwargs):
        return super().__call__(x, target.float(), **kwargs)


LOSSES = {
    'cross_entropy': CELoss,
}
