from networks import NetworkCreator, network_creators
from .pix2pix import GeneratorUNet as Generator
from .ugan import Discriminator, Gradient_Difference_Loss
from .commons import Gradient_Penalty

__all__ = [
    'Generator', 'Discriminator',
    'Gradient_Difference_Loss', 'Gradient_Penalty'
]


@network_creators.register('ugan')
class UGANCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        generator = Generator(cfg['channels'], cfg['channels'])
        discriminator = Discriminator(cfg['channels'])
        net = {'G': generator, 'D': discriminator}
        return net