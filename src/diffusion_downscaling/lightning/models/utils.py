# coding=utf-8
"""Utility helpers for score-based models."""
from __future__ import annotations

import math
from typing import Callable, Dict

import numpy as np

MODEL_REGISTRY: Dict[str, Callable] = {}


def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls

    return decorator


def get_sigmas(config):
    """Return a sigma schedule used for positional embeddings."""
    sigma_min = getattr(config.model, "sigma_min", 0.01)
    sigma_max = getattr(config.model, "sigma_max", 50.0)
    num_scales = getattr(config.model, "num_scales", 1000)

    sigma_min = max(float(sigma_min), 1e-6)
    sigma_max = max(float(sigma_max), sigma_min)
    num_scales = max(int(num_scales), 2)

    return np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_scales))
