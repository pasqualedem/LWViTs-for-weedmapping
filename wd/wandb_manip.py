import wandb
from ruamel.yaml import YAML

from utils.utils import values_to_number
from tqdm import tqdm


def fix_string_param(runs):
    for run in tqdm(runs):
        try:
            params = values_to_number(run.config['in_params'])
            run.config['in_params'] = params
            run.update()
        except KeyError:
            print(f'{run.name} has no in_params')


if __name__ == '__main__':
    param_path = 'resume.yaml'
    with open(param_path, 'r') as param_stream:
        settings = YAML().load(param_stream)
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        print('Query:', query)
        filters = query['filters']
        updated_config = query['updated_config']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        if len(runs) != 0:
            fix_string_param(runs)
