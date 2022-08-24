import argparse
import json

from ruamel.yaml import YAML

from wd.experiment.experiment import experiment
from wd.experiment.resume import resume_run
from wd.utils.utilities import update_collection


parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('--resume_run', required=False, action='store_true',
                    help='Resume the run(s)', default=False)
parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the experiment', default=False)

parser.add_argument('-d', '--dir', required=False, type=str,
                    help='Set the local tracking directory', default=None)

parser.add_argument("--grid", type=int, help="Select the first grid to start from")
parser.add_argument("--run", type=int, help="Select the run in grid to start from")

parser.add_argument('-f', "--filters", type=json.loads, help="Filters to query in the resuming mode")
parser.add_argument('-s', "--stage", type=json.loads, help="Stages to execute in the resuming mode")
parser.add_argument('-p', "--path", type=str, help="Path to the tracking url in the resuming mode")

if __name__ == '__main__':
    args = parser.parse_args()

    track_dir = args.dir
    filters = args.filters
    stage = args.stage
    path = args.path

    if args.resume_run:
        param_path = 'resume.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)
        settings['runs'][0]['filters'] = update_collection(settings['runs'][0]['filters'], filters)
        settings['runs'][0]['stage'] = update_collection(settings['runs'][0]['stage'], stage)
        settings = update_collection(settings, path, key="path")
        resume_run(settings)
    else:
        param_path = 'parameters.yaml'
        with open(param_path, 'r') as param_stream:
            settings = YAML().load(param_stream)
        settings['experiment'] = update_collection(settings['experiment'], args.resume, key='resume')
        settings['experiment'] = update_collection(settings['experiment'], args.grid, key='start_from_grid')
        settings['experiment'] = update_collection(settings['experiment'], args.run, key='start_from_run')
        settings['experiment'] = update_collection(settings['experiment'], track_dir, key='tracking_dir')
        experiment(settings)
