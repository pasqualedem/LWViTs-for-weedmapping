import argparse
import json

from ezdl.utils.utilities import update_collection, load_yaml

parser = argparse.ArgumentParser(description='Train and test models')
parser.add_argument('action',
                    help='Choose the action to perform: '
                         'experiment, resume, resume_run, complete, preprocess, manipulate, app',
                    default="experiment", type=str)
parser.add_argument('--resume', required=False, action='store_true',
                    help='Resume the experiment', default=False)
parser.add_argument('-d', '--dir', required=False, type=str,
                    help='Set the local tracking directory', default=None)
parser.add_argument('-f', '--file', required=False, type=str,
                    help='Set the config file', default=None)
parser.add_argument("--grid", type=int, help="Select the first grid to start from")
parser.add_argument("--run", type=int, help="Select the run in grid to start from")
parser.add_argument("--filters", type=json.loads, help="Filters to query in the resuming mode")
parser.add_argument('-s', "--stage", type=json.loads, help="Stages to execute in the resuming mode")
parser.add_argument('-p', "--path", type=str, help="Path to the tracking url in the resuming mode")

parser.add_argument('--subset', type=str, help="Subset chosen for preprocessing")


def cli():
    args = parser.parse_args()

    track_dir = args.dir
    filters = args.filters
    stage = args.stage
    path = args.path
    action = args.action

    if action == "resume_run":
        from ezdl.experiment.resume import resume_set_of_runs
        param_path = args.file or 'resume.yaml'
        settings = load_yaml(param_path)
        settings['runs'][0]['filters'] = update_collection(settings['runs'][0]['filters'], filters)
        settings['runs'][0]['stage'] = update_collection(settings['runs'][0]['stage'], stage)
        settings = update_collection(settings, path, key="path")
        resume_set_of_runs(settings)
    elif action == 'experiment':
        from ezdl.experiment.experiment import experiment
        param_path = args.file or 'parameters.yaml'
        settings = load_yaml(param_path)
        settings['experiment'] = update_collection(settings['experiment'], args.resume, key='resume')
        settings['experiment'] = update_collection(settings['experiment'], args.grid, key='start_from_grid')
        settings['experiment'] = update_collection(settings['experiment'], args.run, key='start_from_run')
        settings['experiment'] = update_collection(settings['experiment'], track_dir, key='tracking_dir')
        experiment(settings)
    elif action == 'complete':
        from ezdl.experiment.resume import complete_incompleted_runs
        param_path = args.file or 'resume.yaml'
        settings = load_yaml(param_path)
        complete_incompleted_runs(settings)
    elif action == 'manipulation':
        from ezdl.wandb_manip import manipulate
        manipulate()
    elif action in ['app', 'webapp', 'frontend']:
        from ezdl.app import frontend
        frontend()
    else:
        raise ValueError("Action not recognized")


if __name__ == '__main__':
    cli()
