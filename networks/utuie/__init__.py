from networks import NetworkCreator, register_network_creator
from .net import Generator, Discriminator

__all__ = [
    'Generator', 'Discriminator'
]


class UTUIECreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        generator = Generator(**cfg)
        discriminator = Discriminator(**cfg)
        net = {'G': generator, 'D': discriminator}
        return net
    

register_network_creator('utuie', UTUIECreator)