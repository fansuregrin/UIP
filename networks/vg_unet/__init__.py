from networks import NetworkCreator, network_creators
from .vgunet import (
    VGUNet, VGUNet2, VGUNet3, VGUNet4
)

__all__ = ['VGUNet', 'VGUNet2', 'VGUNet3', 'VGUNet4']


@network_creators.register('vgunet')
class VGUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet(**cfg)
    

@network_creators.register('vgunet2')
class VGUNet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet2(**cfg)
    

@network_creators.register('vgunet3')
class VGUNet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet3(**cfg)
    

@network_creators.register('vgunet4')
class VGUNet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet4(**cfg)