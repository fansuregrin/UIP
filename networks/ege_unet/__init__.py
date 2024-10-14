from networks import NetworkCreator, network_creators
from .ege_unet import EGEUNet

__all__ = ['EGEUNet']


@network_creators.register('ege_unet')
class EGEUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return EGEUNet(**cfg)