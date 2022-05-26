import wandb
from ruamel.yaml import YAML

from utils.utils import values_to_number, nested_dict_update
from tqdm import tqdm


def fix_string_param(run):
    try:
        params = values_to_number(run.config['in_params'])
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no in_params')


def update_config(run, config):
    try:
        params = run.config['in_params']
        params = nested_dict_update(params, config)
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no config')


def remove_key(run, config):
    try:
        params = run.config['in_params']
        for key in config:
            params.pop(key)
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no config')


def update_metadata(run, metadata):
    for key, value in metadata.items():
        try:
            run.__dict__[key] = value
        except KeyError:
            print(f'{run.name} has no {key}')


if __name__ == '__main__':
    param_path = 'manip.yaml'
    with open(param_path, 'r') as param_stream:
        settings = YAML().load(param_stream)
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        print('Query:', query)
        filters = query['filters']
        updated_config = query['updated_config']
        updated_metadata = query['updated_meta']
        keys_to_delete = query['keys_to_delete']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        if len(runs) != 0:
            for run in tqdm(runs):
                fix_string_param(run)
                if updated_config is None:
                    print('No config to update')
                else:
                    update_config(run, updated_config)
                if updated_metadata is None:
                    print('No metadata to update')
                else:
                    update_metadata(run, updated_metadata)
                if keys_to_delete is None:
                    print('No keys to delete')
                else:
                    remove_key(run, keys_to_delete)
                run.update()
