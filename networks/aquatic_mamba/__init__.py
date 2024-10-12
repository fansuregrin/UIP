from networks import NetworkCreator, register_network_creator
from .aquatic_mamba import AquaticMambaNet

__all__ = ['AquaticMambaNet']


class AquaticMambaCreator(NetworkCreator):
    def __init__(self):
        super().__init__()
    
    def create_network(cfg):
        return AquaticMambaNet(**cfg)
    

register_network_creator('aqmamba', AquaticMambaCreator)