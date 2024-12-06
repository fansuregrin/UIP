from typing import Dict, Any
from abc import ABC, abstractmethod
from torch import nn

from utils import Registry


class NetworkCreator(ABC):
    """NetworkCreator Abstract Base Class"""
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def create_network(cfg: Dict[str, Any]) -> nn.Module | Dict[str, nn.Module]:
        pass


network_creators = Registry('NetworkCreator')


from networks import (
    aquatic_mamba,
    erd,
    color_enhancer,
    ege_unet,
    fourier_net,
    ugan,
    ultralight_vmunet,
    utuie,
    uvm_net,
    vg_unet,
    vm_unet,
    waternet,
    mimounet,
    mimo_swinT_unet,
    ra_net,
    vit_enhancer,
    msu,
    fcn,
    deeplabv3,
    unet,
    segnet
)

def create_network(cfg: Dict[str, Any]) -> nn.Module | Dict[str, nn.Module]:
    assert 'name' in cfg, "network name is required"
    name = cfg['name']
    net = network_creators.get(name).create_network(cfg)
    return net