from networks import NetworkCreator, network_creators
from .deeplabv3 import DeepLabV3

__all__ = ['DeepLabV3']


@network_creators.register('deeplabv3')
class DeepLabV3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return DeepLabV3(**cfg)