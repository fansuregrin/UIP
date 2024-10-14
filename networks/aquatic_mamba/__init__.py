from networks import NetworkCreator, network_creators
from .aquatic_mamba import AquaticMambaNet

__all__ = ['AquaticMambaNet']


@network_creators.register('aqmamba')
class AquaticMambaCreator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return AquaticMambaNet(**cfg)