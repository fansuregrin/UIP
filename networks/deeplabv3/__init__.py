from networks import NetworkCreator, register_network_creator
from .deeplab import DeepLabV3

__all__ = ['DeepLabV3']


class DeepLabV3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return DeepLabV3(**cfg)
    

register_network_creator('deeplabv3', DeepLabV3Creator)