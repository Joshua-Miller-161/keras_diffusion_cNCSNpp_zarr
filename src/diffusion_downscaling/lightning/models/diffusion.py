import sys
sys.dont_write_bytecode = True
import torch
import torch.nn as nn

from .base import LightningBase
from .karras_diffusion import EDMDenoiser, VPDenoiser


def setup_edm_model(config, score_model, device):
    sigma_data = 0.5
    loss_config = {"buffer_width": config.training.loss_buffer_width}
    loss_model = EDMDenoiser(score_model, sigma_data, device=device)
    score_model = torch.compile(loss_model, options={"triton.cudagraphs": True})
    return score_model, loss_config, loss_model


def setup_vp_model(config, score_model, device):
    loss_config = {"buffer_width": config.training.loss_buffer_width}
    loss_model = VPDenoiser(score_model, device=device)
    score_model = torch.compile(loss_model, options={"triton.cudagraphs": True})
    return score_model, loss_config, loss_model


class LightningDiffusion(LightningBase):
    """
    Lightning level implementation of the diffusion model.

    We rely on the diffusion model to provide the forward pass and loss function,
    which makes this class very light.
    """

    def __init__(self, model, loss_config, optimizer_config, loss_model=None, **kwargs):
        super().__init__(**kwargs)

        self.model = model
        # Some compilation/wrapping paths return callable wrappers that do not
        # expose custom methods like `.loss()` and `.set_buffer_width()`.
        # Keep a reference to an object that definitely owns diffusion loss logic.
        self.loss_model = loss_model if loss_model is not None else self._resolve_loss_model(model)

        self.optimizer_config = optimizer_config
        self.loss_function = self.set_loss_function(loss_config)

    @staticmethod
    def _resolve_loss_model(model):
        # torch.compile can stack wrappers and hide custom methods (e.g. .loss).
        # Walk common wrapper attributes and module children until we find one.
        stack = [model]
        visited = set()

        while stack:
            current = stack.pop()
            if current is None:
                continue

            current_id = id(current)
            if current_id in visited:
                continue
            visited.add(current_id)

            if hasattr(current, "loss"):
                return current

            for attr in ("_orig_mod", "module", "model", "inner_model"):
                maybe_wrapped = getattr(current, attr, None)
                if maybe_wrapped is not None:
                    stack.append(maybe_wrapped)

            if isinstance(current, nn.Module):
                stack.extend(current.children())

        return None

    def forward(self, *x):
        return self.model(*x)

    def set_loss_function(self, loss_config):
        loss_config = dict(loss_config)

        prediction_buffer_width = loss_config.get("buffer_width")
        if prediction_buffer_width is not None and self.loss_model is not None:
            self.loss_model.set_buffer_width(prediction_buffer_width)

        def loss(lightning_module, target, condition):
            if lightning_module.loss_model is None:
                raise AttributeError(
                    "Diffusion model wrapper does not expose `.loss()` and no original "
                    "module with `.loss()` was found. If using torch.compile, keep a "
                    "reference to the original denoiser module or avoid compiling the "
                    "wrapper that owns custom loss methods."
                )
            return lightning_module.loss_model.loss(target, condition)

        return loss
