from networks import NetworkCreator, network_creators
from .ranet import (
    RANet, RANet2, RANet3, RANet4, RANet5, RANet6
)


__all__ = ['RANet', 'RANet2', 'RANet3', 'RANet4', 'RANet5', 'RANet6']


@network_creators.register('ra')
class RANetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet(**cfg)
    

@network_creators.register('ra2')
class RANet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet2(**cfg)
    

@network_creators.register('ra3')
class RANet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet3(**cfg)
    

@network_creators.register('ra4')
class RANet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet4(**cfg)


@network_creators.register('ra5')
class RANet5Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet5(**cfg)
    

@network_creators.register('ra6')
class RANet6Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet6(**cfg)