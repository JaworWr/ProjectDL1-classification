from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
import pandas as pd
from typing import Tuple, Sequence, Dict


DATA_ROOT = "data"


def process_config(config):
    image_shape = config.data.get("image_shape", (256, 256))
    image_shape = tuple(image_shape)
    config.data.image_shape = image_shape
    if "input_shape" not in config.model:
        config.model.input_shape = (*image_shape, 3)


def get_train_valid_loaders(config):
    train_gen = ImageDataGenerator(**config.data.get("augmentation", {}))

    return _get_data_loaders(
        ["train", "valid"],
        train_gen,
        ImageDataGenerator(),
        config.data.get("directory", DATA_ROOT),
        config.data.image_shape,
        config.data.batch_size,
        config.data.get("subset_directory", DATA_ROOT),
        _get_subset_names(config, ["train", "valid"]),
        config.data.get("shuffle", True)
    )


def get_test_loader(config):
    loaders = _get_data_loaders(
        ["test"],
        None,
        ImageDataGenerator(),
        config.data.get("directory", DATA_ROOT),
        config.data.image_shape,
        config.data.batch_size,
        config.data.get("subset_directory", DATA_ROOT),
        _get_subset_names(config, ["test"])
    )
    return loaders["test"]


def _get_subset_names(config, subsets: Sequence[str]):
    def default_name(subset):
        return f"sun_{subset}.csv"

    if "subsets" not in config.data:
        return {subset: default_name(subset) for subset in subsets}
    else:
        return {subset: config.data.subsets.get(subset, default_name(subset)) for subset in subsets}


def _get_data_loaders(
        subsets: Sequence[str],
        train_gen: ImageDataGenerator,
        test_gen: ImageDataGenerator,
        directory: str,
        image_shape: Tuple[int, int],
        batch_size: int,
        subset_dir: str,
        subset_file_names: Dict[str, str],
        shuffle: bool = True
    ):
    loaders = {}
    if test_gen is None:
        test_gen = ImageDataGenerator()

    directory = path.join(directory, "SUN397")

    for subset in subsets:
        df = pd.read_csv(path.join(subset_dir, subset_file_names[subset])).set_index("id")
        data_gen = train_gen if subset == "train" else test_gen
        loaders[subset] = data_gen.flow_from_dataframe(
            dataframe=df,
            directory=directory,
            x_col="path",
            y_col="label",
            target_size=image_shape,
            batch_size=batch_size,
            shuffle=(shuffle and subset == "train")
        )

    if not all(loaders.values()):
        raise RuntimeError("Image data missing. Image directory: " + directory)
    return loaders
