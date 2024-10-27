from abc import ABC, abstractmethod
from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader
from utils import Registry


class DatasetCreator(ABC):
    """Dataset Creator"""
    def __init__(self):
        super().__init__()
    
    @staticmethod
    @abstractmethod
    def create_dataset(cfg: Dict[str, Any]) -> Dataset:
        pass


dataset_creators = Registry('dataset')

from data import dataset


def create_dataset(cfg: Dict[str, Any]) -> Dataset:
    assert 'type' in cfg, "dataset type is required"
    name = cfg['type']
    ds = dataset_creators.get(name).create_dataset(cfg)
    return ds

def create_dataloader(ds, cfg):
    dl = DataLoader(ds, **cfg)
    return dl