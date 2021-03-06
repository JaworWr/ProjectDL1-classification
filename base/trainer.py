from tensorflow.keras import callbacks
import os
from datetime import datetime


class BaseTrainer:
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

        self.callbacks = []
        self.log_dir = None

        if config.trainer.get("tensorboard_enabled", False):
            experiment_name = config.trainer.get("experiment_name",
                                                  datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            directory = config.trainer.get("log_directory", "experiments/logs")
            self.log_dir = os.path.join(directory, experiment_name)
            self._init_tensorboard_callback()

    def init_callbacks(self):
        raise NotImplementedError

    def _init_tensorboard_callback(self):
        tensorboard_callback = callbacks.TensorBoard(log_dir=self.log_dir)
        self.callbacks.append(tensorboard_callback)

    def train(self):
        self.model.fit(
            self.data["train"],
            validation_data=self.data["valid"],
            callbacks=self.callbacks,
            epochs=self.config.trainer.epochs,
            workers=self.config.trainer.get("workers", 1),
        )
