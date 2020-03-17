from base.model import BaseModel
from tensorflow.keras import models, layers, optimizers


class DenseModel(BaseModel):
    def __init__(self, config):
        super().__init__(self, config)
        self.build_model()

    def build_model(self):
        self.model = models.Sequential()
        self.model.append(layers.Flatten(input_shape=self.config.model.input_shape))

        for dim in self.config.model.hidden_dims:
            self.model.append(layers.Dense(dim, activation="relu"))
            if "dropout" in self.config.model:
                self.model.append(layers.Dropout(self.config.model.dropout))

        self.model.append(layers.Dense(397, activation="softmax"))

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.config.training.learning_rate),
            loss="categorical_crossentropy",
            metrics=self.config.training.metrics,
        )