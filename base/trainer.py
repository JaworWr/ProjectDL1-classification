from tensorflow.keras import callbacks
import os
from datetime import datetime


class BaseTrainer:
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

        self.callbacks = []

        if config.training.get("tensorboard_enabled", False):
            experiment_name = config.training.get("experiment_name",
                                                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            directory = config.training.get("log_directory", "experiments/logs")
            self._init_tensorboard_callback(experiment_name, directory)

    def init_callbacks(self):
        pass

    def _init_tensorboard_callback(self, experiment_name: str, directory: str):
        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.join(directory, experiment_name))
        self.callbacks.append(tensorboard_callback)

    def train(self):
        self.model.fit(
            self.data["train"],
            validation_data=self.data["valid"],
            callbacks=self.callbacks,
            epochs=self.config.training.epochs
        )
