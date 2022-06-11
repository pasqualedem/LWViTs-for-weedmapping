from wd.models.regseg import RegSeg48
from wd.models.segnet import SegNet
from wd.models.random import Random
from wd.models.lawin import Lawin, Laweed, DoubleLawin, DoubleLaweed, SplitLawin, SplitLaweed

MODELS = {
    'segnet': SegNet,
    'regseg48': RegSeg48,
    'random': Random,
    'lawin': Lawin,
    'laweed': Laweed,
    'doublelawin': DoubleLawin,
    "doublelaweed": DoubleLaweed,
    'splitlaweed': SplitLaweed
}