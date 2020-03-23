from base.model import BaseModel
from tensorflow.keras import models, layers, optimizers


class DeepModel(BaseModel):
    def build_model(self):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=self.config.model.input_shape))

        for dim in self.config.model.hidden_dims:
            self.model.add(layers.Dense(dim, activation="relu"))
            if "dropout" in self.config.model:
                self.model.add(layers.Dropout(self.config.model.dropout))

        self.model.add(layers.Dense(397, activation="softmax"))

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=float(self.config.model.learning_rate)),
            loss="categorical_crossentropy",
            metrics=self.config.model.metrics,
        )