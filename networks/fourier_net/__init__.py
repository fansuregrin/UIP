from networks import NetworkCreator, register_network_creator
from .fourier_net import FourierNet

__all__ = ['FourierNet']


class FourierNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return FourierNet(**cfg)
    

register_network_creator('fourier_net', FourierNetCreator)