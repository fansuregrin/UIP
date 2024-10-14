from networks import NetworkCreator, network_creators
from .net import Generator, Discriminator

__all__ = [
    'Generator', 'Discriminator'
]


@network_creators.register('utuie')
class UTUIECreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        generator = Generator(**cfg)
        discriminator = Discriminator(**cfg)
        net = {'G': generator, 'D': discriminator}
        return net