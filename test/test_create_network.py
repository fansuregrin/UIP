import unittest
import os
import sys
import yaml
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from networks import create_network


class TestCreateNetwork(unittest.TestCase):
    @staticmethod
    def _load_cfg(cfg_fp: str) -> Dict[str, Any]:
        with open(cfg_fp, 'r') as f:
            cfg = yaml.load(f, yaml.Loader)
        return cfg

    def test_create_aquatic_mamba(self):
        # full
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/aq_mamba_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_erd(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/erd_2down_15res.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_color_enhancer(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ce_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_ege_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ege_unet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_fourier_net(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/fourier_net_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_ugan(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ugan_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_ultralight_vmunet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ultralight_vmunet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_utuie(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/utuie_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_uvm_net(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/uvm_net_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_vgunet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vgunet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_vm_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vmunet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_waternet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/waternet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_mimounet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/mimounet.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_mimo_swinT_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/mimo_1swinT_unet.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_ra_net(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ra_15blocks_2down.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_vit_enhancer1(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vit_enhancer_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_fcn(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/fcn_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_deeplabv3(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/deeplabv3_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/unet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_segnet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/segnet_01.yaml')
        create_network(self._load_cfg(cfg_01))


if __name__ == '__main__':
    unittest.main()