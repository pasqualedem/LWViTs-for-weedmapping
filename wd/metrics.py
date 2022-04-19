from typing import Mapping
from functools import reduce

from torch import Tensor
from torchmetrics import JaccardIndex, AUROC


class AUC(AUROC):
    def update(self, preds: Tensor, target: Tensor) -> None:
        AUROC.update(self, preds.cpu(), target.cpu())


def PerClassAUC(name, code):
    def __init__(self, name, code, *args, **kwargs):
        AUC.__init__(self, num_classes=2, **kwargs)
        self.code = code

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds[:, self.code, ::]
        target = target[:, self.code, ::]
        AUC.update(self, preds, target)

    metric = type(name, (AUC,), {"update": update, "__init__": __init__})
    return metric(name, code)


def metric_instance(name: str, params: dict) -> dict:
    if params.get('discriminator') is not None:
        names = params.pop('discriminator')
        return {
            subname: METRICS[name](subname, code, **params)
            for subname, code in names
        }
    return {name: METRICS[name](**params)}


def metrics_factory(metrics_params: Mapping) -> dict:
    return reduce(lambda a, b: {**a, **b},
                  [
                      metric_instance(name, params)
                      for name, params in metrics_params.items()
                  ]
                  )


METRICS = {
    'jaccard': JaccardIndex,
    'auc': AUC,
    'perclassauc': PerClassAUC,
}
