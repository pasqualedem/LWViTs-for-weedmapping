from typing import Mapping
from functools import reduce

from torch import Tensor
from torchmetrics import JaccardIndex, AUROC, F1Score, Precision, Recall, ConfusionMatrix, ROC
from torchmetrics.functional.classification.roc import _roc_compute
import torch

from copy import deepcopy


class AUC(AUROC):
    def update(self, preds: Tensor, target: Tensor) -> None:
        if self.num_classes is not None and self.num_classes > 2:
            target = target.argmax(dim=1)  # So torchmetric under is multiclass
        AUROC.update(self, preds.cpu(), target.cpu())

    def get_roc(self):
        preds = torch.cat(self.preds, dim=0)
        target = torch.cat(self.target, dim=0)
        if not self.num_classes:
            raise ValueError(f"`num_classes` bas to be positive number, but got {self.num_classes}")
        return _roc_compute(preds, target, self.num_classes, self.pos_label)


def PerClassAUC(name, code):
    def __init__(self, name, code, *args, **kwargs):
        AUC.__init__(self, **kwargs)
        self.code = code

    def update(self, preds: Tensor, target: Tensor) -> None:
        preds = preds[:, self.code, ::].flatten()
        target = target[:, self.code, ::].bool().flatten()
        AUC.update(self, preds, target)

    metric = type(name, (AUC,), {"update": update, "__init__": __init__})
    return metric(name, code)


class WrapJaccard(JaccardIndex):
    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds.argmax(dim=1), target.argmax(dim=1))


class WrapF1(F1Score):
    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds.argmax(dim=1), target.argmax(dim=1))


class WrapPrecision(Precision):
    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds.argmax(dim=1), target.argmax(dim=1))


class WrapRecall(Recall):
    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds.argmax(dim=1), target.argmax(dim=1))


class WrapCF(ConfusionMatrix):
    def update(self, preds: Tensor, target: Tensor) -> None:
        return super().update(preds.argmax(dim=1), target.argmax(dim=1))

    def compute(self) -> Tensor:
        # PlaceHolder value
        return Tensor([1])

    def get_cf(self):
        return super().compute()


def metric_instance(name: str, params: dict) -> dict:
    if params.get('discriminator') is not None:
        params = deepcopy(params)
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
    'jaccard': WrapJaccard,
    'auc': AUC,
    'perclassauc': PerClassAUC,
    'f1': WrapF1,
    'precision': WrapPrecision,
    'recall': WrapRecall,
    'conf_mat': WrapCF
}
