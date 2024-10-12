from networks import NetworkCreator, register_network_creator
from .ege_unet import EGEUNet

__all__ = ['EGEUNet']

class EGEUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return EGEUNet(**cfg)
    

register_network_creator('ege_unet', EGEUNetCreator)