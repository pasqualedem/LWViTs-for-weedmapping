from wd.models.regseg import RegSeg48
from wd.models.segnet import SegNet
from wd.models.random import Random

MODELS = {
    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random
}