from networks import NetworkCreator, register_network_creator
from .segnet import SegNet

__all__ = ['SegNet']


class SegNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return SegNet(**cfg)
    

register_network_creator('segnet', SegNetCreator)