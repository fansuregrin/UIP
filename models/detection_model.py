import os
import time
import sys
import random
import shutil

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
import numpy as np
import yaml
from torch import Tensor
from typing import Union, Dict
from tqdm import tqdm
import cv2


from networks import create_network
from data import create_dataset, create_dataloader
from utils import seed_everything
from third_party.detection.engine import train_one_epoch, evaluate
from third_party.detection._utils import collate_fn
from .base_model import BaseModel
from models import _models


@_models.register('det')
class DetectionModel(BaseModel):
    """Image Enhance Model."""
    def __init__(self, cfg: Dict):
        super().__init__(cfg)

    def setup(self):
        self.name = self.cfg['name']
        self.model_name = self.cfg['model_name']
        self.mode = self.cfg['mode']
        self.device = torch.device(self.cfg.get('device', 'cpu'))
        self._set_network()
        self._set_data()
        
        self.checkpoint_dir = os.path.join(
            self.cfg['ckpt_dir'], self.model_name, self.net_name, self.name)
        self._set_logger()
        self.network_state_dir = os.path.join(self.checkpoint_dir, 'network')
        os.makedirs(self.network_state_dir, exist_ok=True)
        
        if self.mode == 'train':
            self.seed = self.cfg.get('seed', random.randint(0, 100000))
            self.sample_dir = os.path.join(self.checkpoint_dir, 'samples')
            os.makedirs(self.sample_dir, exist_ok=True)
            self.start_epoch = self.cfg['start_epoch']
            self.start_iteration = self.cfg['start_iteration']
            self.num_epochs = self.cfg['num_epochs']
            self.val_interval = self.cfg['val_interval']
            self.ckpt_interval = self.cfg['ckpt_interval']
            # Set optimizers
            self._set_optimizer()
            self.optimizer_state_dir = os.path.join(self.checkpoint_dir, 'optimizer')
            os.makedirs(self.optimizer_state_dir, exist_ok=True)
            # Set lr_scheduler
            self._set_lr_scheduler()
            self.lr_scheduler_state_dir = os.path.join(self.checkpoint_dir, 'lr_scheduler')
            os.makedirs(self.lr_scheduler_state_dir, exist_ok=True)
            self.train_loss = {}
            self.val_loss = {}
            self.train_metrics = {}
            self.val_metrics = {}
            self.tensorboard_log_dir = os.path.join(self.checkpoint_dir, 'tb')
            os.makedirs(self.tensorboard_log_dir, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=self.tensorboard_log_dir)
        elif self.mode == 'test':
            self.result_dir = os.path.join(
                self.cfg['result_dir'], self.model_name, self.net_name, self.name)
            self.epochs = self.cfg['epochs']
            self.test_name = self.cfg['test_name']
            self.load_prefix = self.cfg['load_prefix']
        
    def load_network_state(self, state_name: str):
        state_path = os.path.join(self.network_state_dir, state_name)
        self.network.load_state_dict(torch.load(state_path, weights_only=True))
        if self.logger:
            self.logger.info('Loaded network weights from {}.'.format(
                state_path
            ))

    def load_optimizer_state(self, state_name: str):
        state_path = os.path.join(self.optimizer_state_dir, state_name)
        self.optimizer.load_state_dict(torch.load(state_path, weights_only=True))
        if self.logger:
            self.logger.info('Loaded optimizer state from {}.'.format(
                state_path
            ))

    def load_lr_scheduler_state(self, state_name: str):
        state_path = os.path.join(self.lr_scheduler_state_dir, state_name)
        self.lr_scheduler.load_state_dict(torch.load(state_path, weights_only=True))
        if self.logger:
            self.logger.info('Loaded lr_scheduler state from {}.'.format(
                state_path
            ))

    def _set_network(self):
        with open(self.cfg.get('net_cfg')) as f:
            self.net_cfg = yaml.load(f, yaml.FullLoader)
        self.net_name = self.net_cfg['name']
        self.network = create_network(self.net_cfg)
        if isinstance(self.network, dict):
            for label in self.network:
                self.network[label].to(self.device)
        else:
            self.network.to(self.device)

    def _set_optimizer(self):
        params = self.network.parameters()
        optimizer = self.cfg['optimizer']
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.cfg['lr'])
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.cfg['lr'])
        else:
            assert f"<{optimizer}> is supported!"

    def _set_lr_scheduler(self):
        with open(self.cfg['lr_scheduler_cfg']) as f:
            _cfg = yaml.load(f, yaml.FullLoader)
        if _cfg['name'] == 'none':
            self.lr_scheduler = None
        elif _cfg['name'] == 'step_lr':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, _cfg['step_size'],
                                                          _cfg['gamma'])
        else:
            assert f"<{_cfg['name']}> is supported!"
        self.lr_scheduler_cfg = _cfg

    def _set_logger(self):
        from loguru import logger
        from utils import LOGURU_FORMAT
        logger.remove(0)
        if not self.cfg.get('quite', False):
            logger.add(sys.stdout, format=LOGURU_FORMAT)
        if self.mode == 'train':
            self.log_dir = os.path.join(self.checkpoint_dir, 'logs/train')
        elif self.mode == 'test':
            self.log_dir = os.path.join(self.checkpoint_dir, 'logs/test')
        os.makedirs(self.log_dir, exist_ok=True)
        logger.add(os.path.join(self.log_dir, "{time}.log"), format=LOGURU_FORMAT)
        self.logger = logger
    
    def _set_data(self):
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        with open(self.cfg['ds_cfg']) as f:
            ds_cfg = yaml.load(f, yaml.FullLoader)
        dl_cfg = {
            'batch_size': self.cfg.get('batch_size', 1),
            'shuffle': self.cfg.get('shuffle', True),
            'collate_fn': collate_fn,
            'num_workers': self.cfg.get('num_workers', 0),
            'drop_last': self.cfg.get('drop_last', False)
        }
        if self.mode == 'train':
            train_ds = create_dataset(ds_cfg['train'])
            val_ds = create_dataset(ds_cfg['val'])
            self.train_dl = create_dataloader(train_ds, dl_cfg)
            self.val_dl = create_dataloader(val_ds, dl_cfg)
        elif self.mode == 'test':
            test_ds = create_dataset(ds_cfg['test'])
            dl_cfg['shuffle'] = False
            self.test_dl = create_dataloader(test_ds, dl_cfg)
        else:
            assert False,'invalid mode'

    def _set_loss_fn(self):
        pass

    def _log_training_details(self):
        if self.logger is None: return

        self.logger.info(f"Starting Training Process...")
        for k, v in self.cfg.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("network config details:")
        for k, v in self.net_cfg.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("lr_scheduler config details:")
        for k, v in self.lr_scheduler_cfg.items():
            self.logger.info(f"  {k}: {v}")

    def train(self):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"
        
        seed_everything(self.seed)
        self._log_training_details()
        
        if self.start_epoch > 0:
            load_prefix = self.cfg.get('load_prefix', None)
            if load_prefix:
                state_name = f'{load_prefix}_{self.start_epoch-1}.pth'
                self.load_network_state(state_name)
                self.load_optimizer_state(state_name)
                self.load_lr_scheduler_state(state_name)
            else:
                state_name = f'{self.start_epoch-1}.pth'
                self.load_network_state(state_name)
                self.load_optimizer_state(state_name)
                self.load_lr_scheduler_state(state_name)
        
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            train_one_epoch(self.network, self.optimizer, self.train_dl,
                self.device, epoch, print_freq=self.val_interval)
            # adjust lr
            self.adjust_lr()
            evaluate(self.network, self.val_dl, self.device)
            # save model weights
            if (epoch % self.ckpt_interval == 0) or (epoch == self.start_epoch + self.num_epochs-1):
                self.save_network_weights(epoch)
                self.save_optimizer_state(epoch)
                self.save_lr_scheduler_state(epoch)
    
    def adjust_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
    
    def save_network_weights(self, epoch: int):
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.network_state_dir, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.network_state_dir, "{:d}.pth".format(epoch))
        torch.save(self.network.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved network weights into {}".format(saved_path))

    def save_optimizer_state(self, epoch: int):
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.optimizer_state_dir, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.optimizer_state_dir, "{:d}.pth".format(epoch))
        torch.save(self.optimizer.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved optimizer state into {}".format(saved_path))

    def save_lr_scheduler_state(self, epoch: int):
        if not self.lr_scheduler:
            return
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.checkpoint_dir, 'lr_scheduler', "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.checkpoint_dir, 'lr_scheduler', "{:d}.pth".format(epoch))
        torch.save(self.lr_scheduler.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved lr_shceduler state into {}".format(saved_path))
    
    def test(self):
        pass
    
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            pass
        return parser