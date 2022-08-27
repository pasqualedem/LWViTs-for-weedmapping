from typing import Union

import wandb
from ruamel.yaml import YAML

from utils.utilities import values_to_number, nested_dict_update
from tqdm import tqdm


def fix_string_param(run):
    try:
        params = values_to_number(run.config['in_params'])
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no in_params')
    run.update()


def update_config(run, config):
    try:
        params = run.config['in_params']
        params = nested_dict_update(params, config)
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no config')
    run.update()


def remove_key(run, config):
    try:
        params = run.config['in_params']
        for key in config:
            params.pop(key)
        run.config['in_params'] = params
    except KeyError:
        print(f'{run.name} has no config')
    run.update()


def update_metadata(run, metadata):
    for key, value in metadata.items():
        try:
            run.__dict__[key] = value
        except KeyError:
            print(f'{run.name} has no {key}')
    run.update()


def delete_files(run, files_to_delete: Union[list, str]):
    if type(files_to_delete) == list:
        to_delete = files_to_delete.copy()
        files = run.files()
        for file in files:
            if file.name in to_delete:
                to_delete.pop(to_delete.index(file.name))
                file.delete()
                print(f"{file.name} deleted")
            if len(to_delete) == 0:
                print('All files deleted')
                return
            print(f"{file.name} deleted")

    elif type(files_to_delete) == str:
        to_delete = files_to_delete
        files = run.files()
        for file in files:
            if eval(to_delete)(file):
                file.delete()
                print(f"{file.name} deleted")
    else:
        raise NotImplemented("Use lists or strings")
    print(f"Files remained: {to_delete}")


def delete_artifacts(run, artifacts_to_delete: Union[list, str]):
    if type(artifacts_to_delete) == list:
        to_delete = artifacts_to_delete.copy()
        artifacts = run.logged_artifacts()
        for art in artifacts:
            if art.name in to_delete:
                to_delete.pop(to_delete.index(art.name))
                art.delete()
                print(f"{art.name} deleted")
            if len(to_delete) == 0:
                print('All files deleted')
                return
            print(f"{art.name} deleted")

    elif type(artifacts_to_delete) == str:
        to_delete = artifacts_to_delete
        artifacts = run.logged_artifacts()
        for art in artifacts:
            if eval(to_delete)(art):
                art.delete(delete_aliases=True)
                print(f"{art.name} deleted")
    else:
        raise NotImplemented("Use lists or strings")
    print(f"Artifacts remained: {to_delete}")


def manipulate():
    param_path = 'manip.yaml'
    with open(param_path, 'r') as param_stream:
        settings = YAML().load(param_stream)
    queries = settings['runs']
    path = settings['path']
    for query in queries:
        print('Query:', query)
        filters = query['filters']
        fix_string_params = query['fix_string_params']
        updated_config = query['updated_config']
        updated_metadata = query['updated_meta']
        keys_to_delete = query['keys_to_delete']
        files_to_delete = query['files_to_delete']
        artifacts_to_delete = query['artifacts_to_delete']
        api = wandb.Api()
        runs = api.runs(path=path, filters=filters)
        if len(runs) != 0:
            for run in tqdm(runs):
                print(f"Name: {run.name} Id: {run.id}")
                if fix_string_params:
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
                if files_to_delete is None:
                    print("No files to delete")
                else:
                    delete_files(run, files_to_delete)
                if artifacts_to_delete is None:
                    print("No artifacts to delete")
                else:
                    delete_artifacts(run, artifacts_to_delete)

