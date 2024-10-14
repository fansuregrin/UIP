from networks import NetworkCreator, network_creators
from .color_enhance import ColorEnhancementNet

__all__ = ['ColorEnhancementNet']


@network_creators.register('color_enhancer')
class ColorEnhancerCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ColorEnhancementNet(**cfg)