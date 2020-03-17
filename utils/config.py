import yaml
from dotmap import DotMap


def get_config(path: str):
    with open(path) as f:
        cfg = yaml.safe_load(f)
        cfg = DotMap(cfg)
        return process_config(cfg)


def process_config(cfg):
    image_shape = cfg.data.get("image_shape", (256, 256))
    image_shape = tuple(image_shape)
    cfg.data.image_shape = image_shape
    if "input_shape" not in cfg.model:
        cfg.model.input_shape = (*image_shape, 1)