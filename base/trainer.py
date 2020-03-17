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
            self._init_tensorboard_callback(**self.config.training)

    def init_callbacks(self):
        pass

    def _init_tensorboard_callback(self, experiment_name: str = None, directory: str = "experiments/logs"):
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_callback = callbacks.TensorBoard(log_dir=os.path.join(directory, experiment_name))
        self.callbacks.append(tensorboard_callback)

    def train(self):
        self.model.fit(
            self.data["train"],
            validation_data=self.data["valid"],
            callbacks=self.callbacks,
            epochs=self.config.training.epochs
        )
