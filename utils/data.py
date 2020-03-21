from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
import pandas as pd
from typing import Tuple, Sequence


DATA_ROOT = "data"


def process_config(cfg):
    image_shape = cfg.data.get("image_shape", (256, 256))
    image_shape = tuple(image_shape)
    cfg.data.image_shape = image_shape
    if "input_shape" not in cfg.model:
        cfg.model.input_shape = (*image_shape, 3)


def get_train_valid_loaders(config):
    train_gen = ImageDataGenerator(**config.data.get("augmentation", {}))
    directory = config.data.get("directory", path.join(DATA_ROOT, "SUN397"))

    return _get_data_loaders(
        ["train", "valid"],
        train_gen,
        ImageDataGenerator(),
        directory,
        config.data.image_shape,
        config.data.batch_size
    )


def get_test_loader(config):
    directory = config.data.get("directory", path.join(DATA_ROOT, "SUN397"))
    loaders = _get_data_loaders(
        ["test"],
        None,
        ImageDataGenerator(),
        directory,
        config.data.image_shape,
        config.data.batch_size
    )
    return loaders["test"]


def _get_data_loaders(
        subsets: Sequence[str],
        train_gen: ImageDataGenerator,
        test_gen: ImageDataGenerator,
        directory: str,
        image_shape: Tuple[int, int],
        batch_size: int
    ):
    loaders = {}
    if test_gen is None:
        test_gen = ImageDataGenerator()

    for subset in subsets:
        df = pd.read_csv(path.join(DATA_ROOT, f"sun_{subset}.csv")).set_index("id")
        data_gen = train_gen if subset == "train" else test_gen
        loaders[subset] = data_gen.flow_from_dataframe(
            dataframe=df,
            directory=directory,
            x_col="path",
            y_col="label",
            target_size=image_shape,
            batch_size=batch_size,
            shuffle=(subset == "train")
        )

    if not all(loaders.values()):
        raise RuntimeError("Image data missing. Image directory: " + directory)
    return loaders
