import yaml
from dotmap import DotMap


def get_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
        return DotMap(cfg)
