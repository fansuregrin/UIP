from networks import NetworkCreator, network_creators
from .vit_enhancer import ViTEnhancer1, ViTEnhancer2

__all__ = ['ViTEnhancer1', 'ViTEnhancer2']


@network_creators.register('vit_enhancer1')
class ViTEnhancer1Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ViTEnhancer1(**cfg)
    

@network_creators.register('vit_enhancer2')
class ViTEnhancer2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return ViTEnhancer2(**cfg)