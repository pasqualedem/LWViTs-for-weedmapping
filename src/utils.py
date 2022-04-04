import os
import subprocess

import mlflow
from mlflow.tracking import MlflowClient


def setup_mlflow(exp_name: str):
    """
    Setup mlflow tracking server and experiment
    """
    mlflow_server()
    print('Server started!')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print('URI set!')
    client = MlflowClient()
    exp_info = client.get_experiment_by_name(exp_name)
    exp_id = exp_info.experiment_id if exp_info else MlflowClient().create_experiment(exp_name)
    print('Experiment set')
    return exp_id


def mlflow_server():
    """
    Start mlflow server
    """
    cmd = "mlflow server"
    cmd_env = cmd_env = os.environ.copy()
    child = subprocess.Popen(
        cmd, env=cmd_env, universal_newlines=True, stdin=subprocess.PIPE,
    )