from ezdl.experiment.experiment import Experimenter
from ezdl.utils.utilities import load_yaml


if __name__ == '__main__':
    parameters = load_yaml("parameters.yaml")
    exp = Experimenter()
    exp.calculate_runs(parameters)
    exp.execute_runs()