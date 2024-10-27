import unittest
import os
import sys
import yaml
from typing import Dict, Any

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
from data import create_dataset


class TestCreateDataset(unittest.TestCase):
    @staticmethod
    def _load_cfg(cfg_fp: str) -> Dict[str, Any]:
        with open(cfg_fp, 'r') as f:
            cfg = yaml.load(f, yaml.Loader)
        return cfg
    
    def test_create_paired_img_ds(self):
        cfg = self._load_cfg(os.path.join(ROOT_DIR, 'configs/dataset/lsui.yaml'))
        create_dataset(cfg['train'])

    def test_create_single_img_ds(self):
        cfg = self._load_cfg(os.path.join(ROOT_DIR, 'configs/dataset/u45.yaml'))
        create_dataset(cfg['test'])

    def test_create_seg_ds(self):
        cfg = self._load_cfg(os.path.join(ROOT_DIR, 'configs/dataset/suim.yaml'))
        create_dataset(cfg['train'])

    def test_create_waternet_ds(self):
        cfg = self._load_cfg(os.path.join(ROOT_DIR, 'configs/dataset/lsui.yaml'))
        create_dataset(cfg['waternet_test'])


if __name__ == '__main__':
    unittest.main()