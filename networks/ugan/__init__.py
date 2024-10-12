from networks import NetworkCreator, register_network_creator
from .pix2pix import GeneratorUNet as Generator
from .ugan import Discriminator, Gradient_Difference_Loss
from .commons import Gradient_Penalty

__all__ = [
    'Generator', 'Discriminator',
    'Gradient_Difference_Loss', 'Gradient_Penalty'
]


class UGANCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        generator = Generator(cfg['channels'], cfg['channels'])
        discriminator = Discriminator(cfg['channels'])
        net = {'G': generator, 'D': discriminator}
        return net


register_network_creator('ugan', UGANCreator)