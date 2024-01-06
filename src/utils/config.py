import os
import yaml

def read_params(config_dir: str) -> dict:
    """
    Read parameters from config file
    :param config_dir: directory containing config file
    :return: dictionary of parameters
    """
    with open(os.path.join(config_dir, 'params.yaml'), 'rb') as config:
        params = yaml.safe_load(config)

    return params

