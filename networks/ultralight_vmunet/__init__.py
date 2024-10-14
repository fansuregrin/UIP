from networks import NetworkCreator, network_creators
from .ultralight_vmunet import UltraLight_VM_UNet

__all__ = ['UltraLight_VM_UNet']


@network_creators.register('ultralight_vmunet')
class UltraLightVMUNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return UltraLight_VM_UNet(**cfg)