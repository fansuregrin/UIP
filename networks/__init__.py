from typing import Dict, Any, Union
from abc import ABC, abstractmethod
from torch import nn


class NetworkCreator(ABC):
    """NetworkCreator Abstract Base Class"""
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def create_network(cfg: Dict[str, Any]) -> nn.Module | Dict[str, nn.Module]:
        pass


network_creators: Dict[str, NetworkCreator] = {}

def register_network_creator(name: str, creator: type):
    network_creators[name] = creator

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
    fcn,
    deeplabv3,
    unet,
    segnet
)

def create_network(cfg: Dict[str, Any]) -> nn.Module | Dict[str, nn.Module]:
    assert 'name' in cfg, "network name is required"
    name = cfg['name']
    assert name in network_creators, f'invalid network name: [{name}]'
    net = network_creators[name].create_network(cfg)
    return net
