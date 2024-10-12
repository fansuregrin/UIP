from networks import NetworkCreator, register_network_creator
from .loss import LABLoss, LCHLoss, VGG19_PercepLoss
from .net import Generator, Discriminator

__all__ = [
    'LABLoss', 'LCHLoss', 'VGG19_PercepLoss',
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