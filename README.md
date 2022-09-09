# WeedDetection
## Installation
Recommended to create a Python virtual environment

    pip install -r requirements.txt

## Usage

    python wd.py <args>

**mandatory arguments**
	
	action:	Choose the action to do perform: 
			experiment, resume, resume_run, complete, preprocess, manipulate, app

**optional arguments**:

    -h, --help            show this help message and exit
    --resume              Resume the experiment
    -d DIR, --dir DIR     Set the local tracking directory
    -f FILE, --file FILE  Set the config file
    --grid GRID           Select the first grid to start from
    --run RUN             Select the run in grid to start from
    --filters FILTERS     Filters to query in the resuming mode
    -s STAGE, --stage STAGE
                          Stages to execute in the resuming mode
    -p PATH, --path PATH  Path to the tracking url in the resuming mode
    --subset SUBSET       Subset chosen for preprocessing
		  
### Parameter file
YAML file that contains all parameters necessary to the exepriment to be run.
Some examples can be found in the `params` folder.
