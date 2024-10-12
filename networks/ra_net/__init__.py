from networks import NetworkCreator, register_network_creator
from .ranet import (
    RANet, RANet2, RANet3, RANet4, RANet5, RANet6
)


__all__ = ['RANet', 'RANet2', 'RANet3', 'RANet4', 'RANet5', 'RANet6']


class RANetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet(**cfg)
    

class RANet2Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet2(**cfg)
    

class RANet3Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet3(**cfg)
    

class RANet4Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet4(**cfg)


class RANet5Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet5(**cfg)
    

class RANet6Creator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return RANet6(**cfg)
    

register_network_creator('ra', RANetCreator)
register_network_creator('ra2', RANet2Creator)
register_network_creator('ra3', RANet3Creator)
register_network_creator('ra4', RANet4Creator)
register_network_creator('ra5', RANet5Creator)
register_network_creator('ra6', RANet6Creator)
