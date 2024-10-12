from networks import NetworkCreator, register_network_creator
from .color_enhance import ColorEnhancementNet

__all__ = ['ColorEnhancementNet']


class ColorEnhancerCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ColorEnhancementNet(**cfg)
    

register_network_creator('color_enhancer', ColorEnhancerCreator)