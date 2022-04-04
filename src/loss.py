from torch.nn import CrossEntropyLoss


class CELoss(CrossEntropyLoss):
    def __call__(self, x, target, **kwargs):
        return super().__call__(x, target.float(), **kwargs)


LOSSES = {
    'cross_entropy': CELoss,
}
