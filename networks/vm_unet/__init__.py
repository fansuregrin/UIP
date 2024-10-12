from networks import NetworkCreator, register_network_creator
from .vmunet import VMUNet

__all__ = ['VMUNet']


class VMUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VMUNet(**cfg)
    

register_network_creator('vmunet', VMUNetCreator)