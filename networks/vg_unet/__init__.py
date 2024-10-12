from networks import NetworkCreator, register_network_creator
from .vgunet import (
    VGUNet, VGUNet2, VGUNet3, VGUNet4
)

__all__ = ['VGUNet', 'VGUNet2', 'VGUNet3', 'VGUNet4']


class VGUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet(**cfg)
    

class VGUNet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet2(**cfg)
    

class VGUNet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet3(**cfg)
    

class VGUNet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return VGUNet4(**cfg)
    

register_network_creator('vgunet', VGUNetCreator)
register_network_creator('vgunet2', VGUNet2Creator)
register_network_creator('vgunet3', VGUNet3Creator)
register_network_creator('vgunet4', VGUNet4Creator)