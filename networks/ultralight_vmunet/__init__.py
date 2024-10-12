from networks import NetworkCreator, register_network_creator
from .ultralight_vmunet import UltraLight_VM_UNet

__all__ = ['UltraLight_VM_UNet']


class UltraLightVMUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return UltraLight_VM_UNet(**cfg)
    

register_network_creator('ultralight_vmunet', UltraLightVMUNetCreator)