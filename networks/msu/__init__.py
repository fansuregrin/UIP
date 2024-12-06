from networks import NetworkCreator, network_creators
from .msu import MSU

__all__ = ['MSU']


@network_creators.register('msu')
class MIMOSwinTUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return MSU(**cfg)