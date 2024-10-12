from networks import NetworkCreator, register_network_creator
from .mimo_swinT_unet import (
    MIMOSwinTUNet,
    MIMOSwinTUNet2,
    MIMOSwinTUNet3,
    MIMOSwinTUNet4,
    MIMOSwinTUNet5,
    MIMOSwinTUNet6,
    MIMOSwinTUNet7
)

__all__ = [
    'MIMOSwinTUNet',
    'MIMOSwinTUNet2',
    'MIMOSwinTUNet3',
    'MIMOSwinTUNet4',
    'MIMOSwinTUNet5',
    'MIMOSwinTUNet6',
    'MIMOSwinTUNet7'
]


class MIMOSwinTUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet(**cfg)
    

class MIMOSwinTUNet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet2(**cfg)
    

class MIMOSwinTUNet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet3(**cfg)
    

class MIMOSwinTUNet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet4(**cfg)
    

class MIMOSwinTUNet5Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet5(**cfg)


class MIMOSwinTUNet6Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet6(**cfg)
    

class MIMOSwinTUNet7Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet7(**cfg)
    

register_network_creator('mimo_swinT_unet', MIMOSwinTUNetCreator)
register_network_creator('mimo_swinT_unet2', MIMOSwinTUNet2Creator)
register_network_creator('mimo_swinT_unet3', MIMOSwinTUNet3Creator)
register_network_creator('mimo_swinT_unet4', MIMOSwinTUNet4Creator)
register_network_creator('mimo_swinT_unet5', MIMOSwinTUNet5Creator)
register_network_creator('mimo_swinT_unet6', MIMOSwinTUNet6Creator)
register_network_creator('mimo_swinT_unet7', MIMOSwinTUNet7Creator)