from base.trainer import BaseTrainer
from tensorflow.keras import callbacks

class ReduceLROnPlateauTrainer(BaseTrainer):
    def init_callbacks(self):
        self.callbacks.append(callbacks.ReduceLROnPlateau(**self.config.trainer.reduce_lr_on_plateau))
        if "model_checkpoint" in self.config.trainer:
            self.callbacks.append(callbacks.ModelCheckpoint(
                save_weights_only=True,
                **self.config.model_checkpoint
            ))