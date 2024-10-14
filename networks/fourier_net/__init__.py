from networks import NetworkCreator, network_creators
from .fourier_net import FourierNet

__all__ = ['FourierNet']


@network_creators.register('fourier_net')
class FourierNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return FourierNet(**cfg)