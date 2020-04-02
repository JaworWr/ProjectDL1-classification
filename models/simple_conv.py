from base.model import BaseModel
from tensorflow.keras import models, layers, optimizers, regularizers


class SimpleConvModel(BaseModel):
    def build_model(self):
        self.model = models.Sequential()
        for d in [16, 64, 128, 128]:
            if d == 16:
                self.model.add(layers.Conv2D(d, 3, padding="same", activation="relu",
                                             kernel_regularizer=regularizers.l2(self.config.model.conv_wd),
                                             input_shape=self.config.model.input_shape))
            else:
                self.model.add(layers.Conv2D(d, 3, padding="same", activation="relu",
                                             kernel_regularizer=regularizers.l2(self.config.model.conv_wd)))
            if "conv_dropout" in self.config.model:
                self.model.add(layers.Dropout(self.config.model.conv_dropout))
            self.model.add(layers.Conv2D(d, 3, padding="same", activation="relu",
                                         kernel_regularizer=regularizers.l2(self.config.model.conv_wd)))
            if "conv_dropout" in self.config.model:
                self.model.add(layers.Dropout(self.config.model.conv_dropout))
            self.model.add(layers.MaxPool2D(2))

        self.model.add(layers.Flatten())

        for d in self.config.model.hidden_dims:
            self.model.add(layers.Dense(d, activation="relu",
                                        kernel_regularizer=regularizers.l2(self.config.model.fc_wd)))
            if "dropout" in self.config.model:
                self.model.add(layers.Dropout(self.config.model.dropout))

        self.model.add(layers.Dense(397, activation="softmax",
                                    kernel_regularizer=regularizers.l2(self.config.model.fc_wd)))

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=float(self.config.model.learning_rate)),
            loss="categorical_crossentropy",
            metrics=self.config.model.metrics,
        )
