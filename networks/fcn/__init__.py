from networks import NetworkCreator, register_network_creator
from .fcn import FCN

__all__ = ['FCN']


class FCNCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return FCN(**cfg)
    

register_network_creator('fcn', FCNCreator)