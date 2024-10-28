import yaml
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModel(ABC):
    """Abstract base class (ABC) for models.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the BaseModel class.

        Args:
            cfg: Configurations, a `Dict`.
        """
        super().__init__()
        self.cfg = cfg
        self.setup()

    def setup(self):   
        pass

    @abstractmethod
    def _set_optimizer(self):
        pass

    @abstractmethod
    def _set_lr_scheduler(self):
        pass

    @abstractmethod
    def _set_loss_fn(self):
        pass

    @abstractmethod
    def load_network_state(self):
        pass

    @abstractmethod
    def load_optimizer_state(self):
        pass

    @abstractmethod
    def load_lr_scheduler_state(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def adjust_lr(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def save_network_weights(self):
        pass

    @abstractmethod
    def save_optimizer_state(self):
        pass

    @abstractmethod
    def save_lr_scheduler_state(self):
        pass