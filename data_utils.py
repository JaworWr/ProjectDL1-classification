from keras.preprocessing import image
from os import path
import pandas as pd


DATA_ROOT = "data"


def get_data_loaders(
        train_gen: image.ImageDataGenerator,
        test_gen: image.ImageDataGenerator,
        img_root: str = path.join(DATA_ROOT, "SUN397"),
        **kwargs):
    loaders = {}
    for subset in ["train", "valid", "test"]:
        df = pd.read_csv(path.join(DATA_ROOT, f"sun_{subset}.csv"))
        data_gen = train_gen if subset == "train" else test_gen
        loaders[subset] = data_gen.flow_from_dataframe(
            dataframe=df,
            directory=img_root,
            x_col="path",
            y_col="label",
            **kwargs
        )
    return loaders
