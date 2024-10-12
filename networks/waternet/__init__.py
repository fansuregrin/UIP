from networks import NetworkCreator, register_network_creator
from .waternet import WaterNet

__all__ = ['WaterNet']


class WaterNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return WaterNet(**cfg)
    

register_network_creator('waternet', WaterNetCreator)