from networks import NetworkCreator, network_creators
from .mimo_unet import MIMOUNet, MIMOUNetPlus

__all__ = ['MIMOUNet', 'MIMOUNetPlus']


@network_creators.register('mimounet')
class MIMOUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOUNet(**cfg)
    

@network_creators.register('mimounet_plus')
class MIMOUNetPlusCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOUNetPlus(**cfg)