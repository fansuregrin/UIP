from networks import NetworkCreator, network_creators
from .segnet import SegNet

__all__ = ['SegNet']


@network_creators.register('segnet')
class SegNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return SegNet(**cfg)