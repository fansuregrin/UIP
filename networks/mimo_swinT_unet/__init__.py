from networks import NetworkCreator, network_creators
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


@network_creators.register('mimo_swinT_unet')
class MIMOSwinTUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet(**cfg)
    

@network_creators.register('mimo_swinT_unet2')
class MIMOSwinTUNet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet2(**cfg)
    

@network_creators.register('mimo_swinT_unet3')
class MIMOSwinTUNet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet3(**cfg)
    

@network_creators.register('mimo_swinT_unet4')
class MIMOSwinTUNet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet4(**cfg)
    

@network_creators.register('mimo_swinT_unet5')
class MIMOSwinTUNet5Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet5(**cfg)


@network_creators.register('mimo_swinT_unet6')
class MIMOSwinTUNet6Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet6(**cfg)
    

@network_creators.register('mimo_swinT_unet7')
class MIMOSwinTUNet7Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOSwinTUNet7(**cfg)