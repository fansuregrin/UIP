from networks import NetworkCreator, network_creators
from .unet import UNet

__all__ = ['UNet']


@network_creators.register('unet')
class UNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return UNet(**cfg)