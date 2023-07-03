# ViTs for WeedMapping

Github repo for the paper "Weed Mapping in Multispectral Drone Imagery using Lightweight Vision Transformers"

## Installation
Recommended to create a Python virtual environment. Install the requirements

    pip install -r requirements.txt

## Experiment reproduction

### Download the WeedMap dataset
    py wd.py download

Or directly download the tiles from [here](https://projects.asl.ethz.ch/datasets/doku.php?id=weedmap:remotesensing2018weedmap)

Directory tree should be like this:


    dataset/
    ├─ raw/
    │  ├─ RedEdge/
    │  ├─ Sequoia/


### Preprocessing

    py wd.py preprocess --subset RedEdge
    py wd.py preprocess --subset Sequoia

### Data augmentation

    py wd.py augment --subset RedEdge
    py wd.py augment --subset Sequoia

### Training and Testing

    python wd.py experiment --file params/RedEdge/SplitLawin.yaml

Refer to the params folder for the different experiments made
