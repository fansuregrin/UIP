from networks import NetworkCreator, register_network_creator
from .unet import UNet

__all__ = ['UNet']


class UNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return UNet(**cfg)
    

register_network_creator('unet', UNetCreator)