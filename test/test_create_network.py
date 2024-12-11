import unittest
import os
import sys
from typing import Dict, Any

import yaml
import torch

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
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/aq_mamba/aq_mamba_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_erd(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/erd/erd_2down_15res.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_erd2(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/erd2/erd2_3down_8res.yaml')
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
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vagunet/vgunet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_vm_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vmunet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_waternet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/waternet/waternet_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_mimounet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/mimounet.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_mimo_swinT_unet(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/mimo_swint_unet/mimo_1swinT_unet.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_ra_net(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/ra/ra_15blocks_2down.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_vit_enhancer1(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/vit_enhancer_01.yaml')
        create_network(self._load_cfg(cfg_01))

    def test_create_fcn(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/fcn/fcn_01.yaml')
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

    def test_create_msu(self):
        cfg_01 = os.path.join(ROOT_DIR, 'configs/network/msu/msu_8res_4swinT.yaml')
        create_network(self._load_cfg(cfg_01))


class TestMSU(unittest.TestCase):
    @staticmethod
    def _create_and_run(cfg_fp: str):
        with open(cfg_fp, 'r') as f:
            cfg = yaml.load(f, yaml.Loader)
        net = create_network(cfg).to('cuda')
        x = torch.randn((4, 3, 256, 256)).to('cuda')
        y = net(x)
        return y

    def test_full_model(self):
        y = self._create_and_run(
            os.path.join(ROOT_DIR, 'configs/network/msu/msu_8res_4swinT.yaml'))
        assert y[1].shape == (4, 3, 256, 256)
        assert y[0.5].shape == (4, 3, 128, 128)
        assert y[0.25].shape == (4, 3, 64, 64)

    def test_wo_fhm(self):
        y = self._create_and_run(
            os.path.join(ROOT_DIR, 'configs/network/msu/msu_8res_4swinT_02.yaml'))
        assert y[1].shape == (4, 3, 256, 256)
        assert y[0.5].shape == (4, 3, 128, 128)
        assert y[0.25].shape == (4, 3, 64, 64)

    def test_one_scale(self):
        y = self._create_and_run(
            os.path.join(ROOT_DIR, 'configs/network/msu/msu_8res_4swinT_03.yaml'))
        assert y[1].shape == (4, 3, 256, 256)
    
    def test_two_scale(self):
        y = self._create_and_run(
            os.path.join(ROOT_DIR, 'configs/network/msu/msu_8res_4swinT_04.yaml'))
        assert y[1].shape == (4, 3, 256, 256)
        assert y[0.5].shape == (4, 3, 128, 128)

if __name__ == '__main__':
    unittest.main()