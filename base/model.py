class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None

    # save function that saves the checkpoint in the path defined in the config file
    def save(self, checkpoint_path):
        if self.model is None:
            raise RuntimeError("You have to build the model before saving it.")

        print("Saving model...")
        self.model.save_weights(checkpoint_path)
        print("Model saved")

    # load latest checkpoint from the experiment path defined in the config file
    def load(self, checkpoint_path):
        if self.model is None:
            raise RuntimeError("You have to build the model before loading it.")

        print(f"Loading model checkpoint {checkpoint_path}...")
        self.model.load_weights(checkpoint_path)
        print("Model loaded")

    def build_model(self):
        raise NotImplementedError