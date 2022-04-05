from torchmetrics import JaccardIndex, AUROC


METRICS = {
    'jaccard': JaccardIndex,
    'auc': AUROC
}