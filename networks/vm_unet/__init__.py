from networks import NetworkCreator, network_creators
from .vmunet import VMUNet

__all__ = ['VMUNet']


@network_creators.register('vmunet')
class VMUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VMUNet(**cfg)