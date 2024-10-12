from networks import NetworkCreator, register_network_creator
from .vit_enhancer import ViTEnhancer1, ViTEnhancer2

__all__ = ['ViTEnhancer1', 'ViTEnhancer2']


class ViTEnhancer1Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ViTEnhancer1(**cfg)
    

class ViTEnhancer2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ViTEnhancer2(**cfg)
    

register_network_creator('vit_enhancer1', ViTEnhancer1Creator)
register_network_creator('vit_enhancer2', ViTEnhancer2Creator)