from networks import NetworkCreator, register_network_creator
from .mimo_unet import MIMOUNet, MIMOUNetPlus

__all__ = ['MIMOUNet', 'MIMOUNetPlus']


class MIMOUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOUNet(**cfg)
    

class MIMOUNetPlusCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MIMOUNetPlus(**cfg)
    

register_network_creator('mimounet', MIMOUNetCreator)
register_network_creator('mimounet_plus', MIMOUNetPlusCreator)