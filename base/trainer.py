class BaseTrainer:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

        self.callbacks = []

    def init_callbacks(self):
        pass

    def train(self):
        self.model.fit(
            self.data["train"],
            validation_data=self.data["valid"],
            callbacks=self.callbacks,
            **self.config.trainer
        )
