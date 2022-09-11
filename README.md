# ViTs for WeedMapping

Different experiments for the adaptation of the Lawin transformer architecture
for the Weed Mapping task, on the WeedMap dataset.

## Installation
Recommended to create a Python virtual environment. Install the requirements

    pip install -r requirements.txt

## Usage

### Preprocessing

    python wd/preprocessing.py <subset>

where subset can be SEQUOIA or REDEDGE

### Training and Testing

    python ezdl experiment

By default, parameters.yaml will be used. All the experiment yaml files are in the params folders, for example:

    python ezdl experiment --file params/Sequoia/Focal.yaml
