"""Trainer backends: select the underlying Gaussian-Splatting codebase."""
from .trainer_backends import (
    FasterGSBackend,
    ThreeDGSBackend,
    TrainerBackend,
    available_backends,
    load_backend,
)

__all__ = [
    "TrainerBackend",
    "ThreeDGSBackend",
    "FasterGSBackend",
    "load_backend",
    "available_backends",
]
