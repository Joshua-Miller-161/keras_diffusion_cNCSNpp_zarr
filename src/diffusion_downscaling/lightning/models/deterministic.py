import torch.nn as nn
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import numpy as np

from .base import LightningBase


class LightningDeterministic(LightningBase):
    """
    Regression-based model for deterministic downscaling.

    This model is a simple regression model that takes conditioning data as input
    and outputs a downscaling of the input data. The model is trained using a
    specified loss function, and the model architecture is provided by the user.
    """
    def __init__(self, model, loss_config, optimizer_config, **kwargs):
        super().__init__(**kwargs)

        self.model = model

        self.optimizer_config = optimizer_config
        self.loss_function = self.set_loss_function(loss_config)

    def forward(self, *x):
        return self.model(*x)

    def set_loss_function(self, loss_config):
        loss_config = dict(loss_config)
        loss_type = loss_config.pop("loss_type")
        return self.create_loss_function(loss_type)

    def create_loss_function(self, loss_type):
        """
        Could be extended if more loss functions or training schemes are needed.
        """
        if loss_type == "mse":

            def loss(model, outputs, conditioning):
                return F.mse_loss(model(None, conditioning), outputs).mean()

            return loss
        else:
            raise NotImplementedError(f"{loss_type} loss type not implemented.")
