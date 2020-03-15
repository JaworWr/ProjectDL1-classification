from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import path
import pandas as pd
from typing import Tuple


DATA_ROOT = "data"


def get_data_loaders(
        train_gen: ImageDataGenerator,
        test_gen: ImageDataGenerator = None,
        directory: str = path.join(DATA_ROOT, "SUN397"),
        image_shape: Tuple[int, int] = (256, 256),
        **kwargs):
    loaders = {}
    if test_gen is None:
        test_gen = ImageDataGenerator()

    for subset in ["train", "valid", "test"]:
        df = pd.read_csv(path.join(DATA_ROOT, f"sun_{subset}.csv")).set_index("id")
        data_gen = train_gen if subset == "train" else test_gen
        loaders[subset] = data_gen.flow_from_dataframe(
            dataframe=df,
            directory=directory,
            x_col="path",
            y_col="label",
            target_size=image_shape,
            **kwargs
        )

    if not all(loaders.values()):
        raise RuntimeError("Image data missing. Image directory: " + directory)
    return loaders
