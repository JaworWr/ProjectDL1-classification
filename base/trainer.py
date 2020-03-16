class BaseTrainer:
    def __init__(self, config, model, data):
        self.config = config
        self.model = model
        self.data = data

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
