import os
import time
import random
import shutil
import sys
from typing import Dict

import yaml
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
import numpy as np
from tqdm import tqdm
from torchvision.utils import draw_segmentation_masks
from kornia.losses import DiceLoss
# from kornia.metrics import mean_iou
from torchmetrics.functional.segmentation import mean_iou

from .base_model import BaseModel
from data import create_dataloader, create_dataset
from networks import create_network
from models import _models
from utils import seed_everything


@_models.register('seg')
class SegModel(BaseModel):
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
            # Set Loss function
            self._set_loss_fn()
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

    def _set_optimizer(self):
        params = self.network.parameters()
        optimizer = self.cfg['optimizer']
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                params, lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                params, lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
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

    def _set_loss_fn(self):
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.dice_loss_fn = DiceLoss().to(self.device)
        self.lambda_ce = self.cfg['lambda_ce']
        self.lambda_dice = self.cfg['lambda_dice']

    def _set_data(self):
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        with open(self.cfg['ds_cfg']) as f:
            ds_cfg = yaml.load(f, yaml.FullLoader)
        self.cfg['shuffle'] = True if self.mode == 'train' else False
        dl_cfg = {
            'batch_size': self.cfg.get('batch_size', 1),
            'shuffle': self.cfg['shuffle'],
            'num_workers': self.cfg.get('num_workers', 0),
            'drop_last': self.cfg.get('drop_last', False)
        }
        self.classes = ds_cfg['classes']
        if self.mode == 'train':
            train_ds = create_dataset(ds_cfg['train'])
            val_ds = create_dataset(ds_cfg['val'])
            self.train_dl = create_dataloader(train_ds, dl_cfg)
            self.val_dl = create_dataloader(val_ds, dl_cfg)
        elif self.mode == 'test':
            test_ds = create_dataset(ds_cfg['test'])
            self.test_dl = create_dataloader(test_ds, dl_cfg)
        else:
            assert False,'invalid mode'

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

    def load_network_state(self, state_name: str):
        state_path = os.path.join(self.network_state_dir, state_name)
        self.network.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded network weights from [{}]'.format(
                state_path
            ))

    def load_optimizer_state(self, state_name: str):
        state_path = os.path.join(self.optimizer_state_dir, state_name)
        self.optimizer.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded optimizer state from {}.'.format(
                state_path
            ))

    def load_lr_scheduler_state(self, state_name: str):
        state_path = os.path.join(self.lr_scheduler_state_dir, state_name)
        self.lr_scheduler.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded lr_scheduler state from {}.'.format(
                state_path
            ))

    def train(self):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"
        seed_everything(self.seed)

        if not self.logger is None:
            self.logger.info(f"Starting Training Process...")
            for k, v in self.cfg.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info("network config details:")
            for k, v in self.net_cfg.items():
                self.logger.info(f"  {k}: {v}")
            self.logger.info("lr_scheduler config details:")
            for k, v in self.lr_scheduler_cfg.items():
                self.logger.info(f"  {k}: {v}")

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
        
        iteration_index = self.start_iteration
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            for i, batch in enumerate(self.train_dl):
                # train one batch
                self.train_one_batch(batch)
                # validation
                if (iteration_index % self.val_interval == 0) or (i == len(self.train_dl)-1):
                    val_batch = next(iter(self.val_dl))
                    self.validate_one_batch(val_batch, iteration_index)
                    self.write_tensorboard(iteration_index)
            
                    if self.logger:
                        self.logger.info(
                            "[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] [train_loss: {:.3f}, val_loss: {:.3f}]".format(
                            iteration_index, self.optimizer.param_groups[0]['lr'],
                            epoch, self.start_epoch + self.num_epochs-1, i, len(self.train_dl)-1,
                            self.train_loss['ce'].item(), self.val_loss['ce'].item()
                        ))
                iteration_index += 1
            # adjust lr
            self.adjust_lr()
            # save model weights
            if (epoch % self.ckpt_interval == 0) or (epoch == self.start_epoch + self.num_epochs-1):
                self.save_network_weights(epoch)
                self.save_optimizer_state(epoch)
                self.save_lr_scheduler_state(epoch)

    def train_one_batch(self, input_: Dict):
        inp_imgs = input_['img'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        self.optimizer.zero_grad()
        self.network.train()
        pred_masks = self.network(inp_imgs)
        self._calculate_loss(ref_masks, pred_masks)
        self._calculate_metrics(ref_masks, pred_masks)
        self.train_loss['total'].backward()
        self.optimizer.step()
    
    def adjust_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
    
    def write_tensorboard(self, iteration):
        for loss_name in self.train_loss.keys():
            self.tb_writer.add_scalars(f'loss/{loss_name}',
                                       {
                                           'train': self.train_loss[loss_name],
                                           'val': self.val_loss[loss_name],
                                       },
                                       iteration)
        for metric_name in self.train_metrics.keys():
            self.tb_writer.add_scalars(f'metrics/{metric_name}',
                                       {
                                           'train': self.train_metrics[metric_name],
                                           'val': self.val_metrics[metric_name],
                                       },
                                       iteration)
    
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
            saved_path = os.path.join(self.lr_scheduler_state_dir, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.lr_scheduler_state_dir, "{:d}.pth".format(epoch))
        torch.save(self.lr_scheduler.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved lr_shceduler state into {}".format(saved_path))

    def _calculate_loss(self, ref_masks, pred_masks, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['ce'] = self.ce_loss_fn(pred_masks, ref_masks)
        loss['dice'] = self.dice_loss_fn(
            F.softmax(pred_masks, dim=1),
            ref_masks)
        loss['total'] = loss['ce'] * self.lambda_ce +\
                        loss['dice'] * self.lambda_dice
        
    def _calculate_metrics(self, ref_masks, pred_masks, train=True):
        metrics = self.train_metrics if train else self.val_metrics
        # metrics['mIoU'] = mean_iou(
        #     F.softmax(pred_masks, dim=1).argmax(1),
        #     ref_masks,
        #     len(self.classes)).mean(1).mean(0)
        metrics['mIoU'] = mean_iou(pred_masks.argmax(1), ref_masks, 
                                   len(self.classes), input_format='index').mean()

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['img'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        with torch.no_grad():
            pred_masks = self.network(inp_imgs)
            self._calculate_loss(ref_masks, pred_masks, train=False)
            self._calculate_metrics(ref_masks, pred_masks, train=False)
            full_img = self._gen_comparison_img(inp_imgs.cpu(), pred_masks.cpu(), ref_masks.cpu())
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)
    
    def test(self):
        if not self.logger is None:
            self.logger.info(f"Starting Training Process...")
            for k, v in self.cfg.items():
                self.logger.info(f"{k}: {v}")
            self.logger.info("network config details:")
            for k, v in self.net_cfg.items():
                self.logger.info(f"  {k}: {v}")
        
        for epoch in self.epochs:
            self.test_one_epoch(epoch, self.test_name, self.load_prefix)

    def test_one_epoch(self, epoch: int, test_name: str, load_prefix=None):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"
        
        weights_name = f"{load_prefix}_{epoch}" if load_prefix else f"{epoch}"
        self.load_network_state(f"{weights_name}.pth")
        result_dir = os.path.join(self.result_dir, test_name, weights_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)
        os.makedirs(os.path.join(result_dir, 'paired'))

        columns = ['img_name', 'time', 'mIoU']
        columns += [f"mIoU_{cls['symbol']}" for cls in self.classes]
        record = pd.DataFrame(columns=columns)

        batch_idx = 1
        for batch in tqdm(self.test_dl):
            inp_imgs = batch['img'].to(self.device)
            ref_masks = batch['mask'].to(self.device) if 'mask' in batch else None
            img_names = batch['img_name']
            batch_size = len(inp_imgs)
            
            with torch.no_grad():
                self.network.eval()
                t_start = time.time()
                pred_masks = self.network(inp_imgs)
                # consumed time
                t_elapse = (time.time() - t_start)
            
            # global mIoU & mIoU per class
            if not ref_masks is None:
                mIoU = mean_iou(pred_masks.softmax(1).argmax(1),
                    ref_masks, len(self.classes), input_format='index')
                mIoU_per_class = mean_iou(pred_masks.softmax(1).argmax(1), ref_masks,
                    len(self.classes), per_class=True, input_format='index')
            
            for i in range(batch_size):
                img_name = img_names[i]
                row = {'img_name': img_name, 'time': t_elapse/batch_size}
                row['mIoU'] = mIoU[i].cpu().item()
                for cls in self.classes:
                    row[f"mIoU_{cls['symbol']}"] = mIoU_per_class[i][cls['id']].cpu().item()
                record.loc[len(record)] = row

            # visualized results
            full_img = self._gen_comparison_img(inp_imgs, pred_masks, ref_masks)
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_dir, 'paired', f'{batch_idx:06d}.png'), full_img)
            batch_idx += 1

        record.loc[len(record)] = record.select_dtypes(include='number').mean()
        record.loc[len(record)-1, 'img_name'] = 'average'
        record.to_csv(os.path.join(result_dir, 'record.csv'), index=False)

        if self.logger:
            self.logger.info(f'saved result into [{result_dir}]')
    
    def _gen_comparison_img(self, imgs: Tensor, pred_masks: Tensor, ref_masks = None):
        colors = ['#'+clz['color'] for clz in self.classes]
        imgs_with_pred_mask = []
        imgs_with_ref_mask = []
        imgs = imgs.to(torch.device('cpu'))
        pred_masks = pred_masks.to(torch.device('cpu'))
        if ref_masks is not None:
            ref_masks = ref_masks.to(torch.device('cpu'))
            for img, pred_mask, ref_mask in zip(imgs, pred_masks, ref_masks):
                img = (img * 255).to(torch.uint8)
                normalized_pred_mask = F.softmax(pred_mask, dim=0)
                boolean_pred_mask = torch.stack(
                    [(normalized_pred_mask.argmax(0) == i) for i in range(len(self.classes))]
                )
                img_with_pred_mask = draw_segmentation_masks(img, boolean_pred_mask, alpha=0.5, colors=colors)
                imgs_with_pred_mask.append(img_with_pred_mask.cpu().numpy().transpose(1,2,0))

                # normalized_ref_mask = F.softmax(ref_mask, dim=0)
                # boolean_ref_mask = torch.stack(
                #     [(normalized_ref_mask.argmax(0) == i) for i in range(len(self.classes))]
                # )
                boolean_ref_mask = torch.stack([ref_mask == i for i in range(len(self.classes))])
                img_with_ref_mask = draw_segmentation_masks(img, boolean_ref_mask, alpha=0.5, colors=colors)
                imgs_with_ref_mask.append(img_with_ref_mask.cpu().numpy().transpose(1,2,0))
            imgs_with_pred_mask = np.concatenate(imgs_with_pred_mask, axis=1)
            imgs_with_ref_mask = np.concatenate(imgs_with_ref_mask, axis=1)
            full_img = np.concatenate((imgs_with_pred_mask, imgs_with_ref_mask), axis=0)
        else:
            for img, pred_mask in zip(imgs, pred_masks):
                img = (img * 255).to(torch.uint8)
                normalized_pred_mask = F.softmax(pred_mask, dim=0)
                boolean_pred_mask = torch.stack(
                    [(normalized_pred_mask.argmax(0) == i) for i in range(len(self.classes))]
                )
                img_with_pred_mask = draw_segmentation_masks(img, boolean_pred_mask, alpha=0.5, colors=colors)
                imgs_with_pred_mask.append(img_with_pred_mask.cpu().numpy().transpose(1,2,0))
            imgs_with_pred_mask = np.concatenate(imgs_with_pred_mask, axis=1)
            full_img = img_with_pred_mask
        return full_img
    
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument('--weight_decay', type=float, default=0.0)
            parser.add_argument('--lambda_ce', type=float, default=1.0)
            parser.add_argument('--lambda_dice', type=float, default=1.0)
        return parser