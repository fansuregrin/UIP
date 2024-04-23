import torch
from abc import ABC
from typing import Union, Dict, Any

from networks import create_network


class BaseModel(ABC):
    """Abstract base class (ABC) for models.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the BaseModel class.

        Args:
            cfg: Configurations, a `Dict`.
        """
        self.device = cfg.get('device', torch.device('cpu'))
        self.logger =  cfg.get('logger', None)
        self.net_cfg = cfg['net_cfg']
        self.mode = cfg['mode']
        self.cfg = cfg
        self.setup()

    def setup(self):
        """Setup the model.
        """
        self.network = create_network(self.net_cfg)
        if isinstance(self.network, dict):
            for label in self.network:
                self.network[label].to(self.device)
        else:
            self.network.to(self.device)
        if self.mode == 'train':
            self.tb_writer = self.cfg['tb_writer']
            self.sample_dir = self.cfg['sample_dir']
            self.checkpoint_dir = self.cfg['checkpoint_dir']
            self.name = self.cfg['name']
            self.start_epoch = self.cfg['start_epoch']
            self.start_iteration = self.cfg['start_iteration']
            self.num_epochs = self.cfg['num_epochs']
            self.val_interval = self.cfg['val_interval']
            self.ckpt_interval = self.cfg['ckpt_interval']
        else:
            assert f"{self.mode} is not supported!"