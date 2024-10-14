from networks import NetworkCreator, network_creators
from .waternet import WaterNet

__all__ = ['WaterNet']


@network_creators.register('waternet')
class WaterNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return WaterNet(**cfg)