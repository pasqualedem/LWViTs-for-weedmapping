import os
import zipfile
from pathlib import Path

import wget
import shutil


def progress_bar(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))


DATA_ROOT = 'dataset/raw'

RED_EDGE = "http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Tiles/RedEdge.zip"
SEQUOIA = "http://robotics.ethz.ch/~asl-datasets/2018-weedMap-dataset-release/Tiles/Sequoia.zip"

REDEDGE_ZIP = 'RedEdge.zip'
SEQUOIA_ZIP = 'Sequoia.zip'


def download():
    os.makedirs(DATA_ROOT, exist_ok=True)
    wget.download(RED_EDGE, REDEDGE_ZIP, progress_bar)
    wget.download(SEQUOIA, SEQUOIA_ZIP, progress_bar)
    with zipfile.ZipFile(REDEDGE_ZIP, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_ROOT, 'RedEdge'))
    with zipfile.ZipFile(SEQUOIA_ZIP, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_ROOT, 'Sequoia'))
    os.remove(REDEDGE_ZIP)
    os.remove(SEQUOIA_ZIP)

    shutil.rmtree(os.path.join(DATA_ROOT, 'RedEdge', "__MACOSX"))
    shutil.rmtree(os.path.join(DATA_ROOT, 'Sequoia', "__MACOSX"))
    for path in Path('dataset').rglob('.DS_Store'):
        os.remove(path)