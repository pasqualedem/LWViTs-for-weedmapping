import os
import subprocess
from typing import Any, Optional, Mapping

import mlflow
import torch
from mlflow.tracking import MlflowClient


def setup_mlflow(exp_name: str, description: str) -> str:
    """
    Setup mlflow tracking server and experiment
    """
    mlflow_server()
    print('Server started!')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    print('URI set!')
    client = MlflowClient()
    exp_info = client.get_experiment_by_name(exp_name)
    exp_id = exp_info.experiment_id if exp_info else \
        MlflowClient().create_experiment(exp_name, tags={'mlflow.note.content': description})
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


class MLRun(MlflowClient):
    def __init__(self, exp_name: str, description: str):
        super().__init__()
        exp_id = setup_mlflow(exp_name, description)
        self.run = self.create_run(experiment_id=exp_id)

    def log_params(self, params: Mapping):
        for k, v in params.items():
            self.log_param(k, v)

    def log_metrics(self, metrics: Mapping):
        for k, v in metrics.items():
            self.log_metric(k, v)

    def log_param(self, key: str, value: Any, run_id: str = None) -> None:
        if run_id is None:
            run_id = self.run.info.run_id
        super().log_param(run_id, key, value)

    def log_metric(self, key: str, value: Any, run_id: str = None,
                   timestamp: Optional[int] = None,
                   step: Optional[int] = None) -> None:

        value = value.item() if type(value) == torch.Tensor else value

        if run_id is None:
            run_id = self.run.info.run_id
        super().log_metric(run_id, key, value)

