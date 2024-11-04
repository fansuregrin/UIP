import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
import sys
import yaml
import random
from torch import Tensor
from typing import Union, Dict
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.metrics import psnr, ssim
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import cv2
import shutil

from networks import create_network
from data import create_dataset, create_dataloader
from utils import seed_everything
from .base_model import BaseModel
from losses import (
    L1CharbonnierLoss, FourDomainLoss, 
    EdgeLoss, FourDomainLoss2, FourDomainLoss3,
    S3IM, ContrastLoss
)
from models import _models


@_models.register('ie')
class ImgEnhanceModel(BaseModel):
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
        
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
    
    def _set_data(self):
        self.train_dl = None
        self.val_dl = None
        self.test_dl = None
        with open(self.cfg['ds_cfg']) as f:
            ds_cfg = yaml.load(f, yaml.FullLoader)
        dl_cfg = {
            'batch_size': self.cfg.get('batch_size', 1),
            'shuffle': self.cfg.get('shuffle', True),
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

    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr']
        
    def _calculate_metrics(self, ref_imgs, pred_imgs, train=True):
        metrics = self.train_metrics if train else self.val_metrics
        metrics['psnr'] = psnr(pred_imgs, ref_imgs, 1.0)
        metrics['ssim'] = ssim(pred_imgs, ref_imgs, 11).mean()

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
                    if not self.logger is None:
                        self.logger.info(
                            "[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] "
                            "[train_loss: {:.3f}, val_loss: {:.3f}]".format(
                                iteration_index, self.optimizer.param_groups[0]['lr'],
                                epoch, self.start_epoch + self.num_epochs-1, i, len(self.train_dl)-1,
                                self.train_loss['total'].item(), self.val_loss['total'].item()
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
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        self.optimizer.zero_grad()
        self.network.train()
        pred_imgs = self.network(inp_imgs)
        self._calculate_loss(ref_imgs, pred_imgs)
        self.train_loss['total'].backward()
        self.optimizer.step()
        self._calculate_metrics(ref_imgs, pred_imgs)
    
    def adjust_lr(self):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
    
    def write_tensorboard(self, iteration: int):
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
            saved_path = os.path.join(self.checkpoint_dir, 'lr_scheduler', "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.checkpoint_dir, 'lr_scheduler', "{:d}.pth".format(epoch))
        torch.save(self.lr_scheduler.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved lr_shceduler state into {}".format(saved_path))

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        with torch.no_grad():
            pred_imgs = self.network(inp_imgs)
            
            self._calculate_loss(ref_imgs, pred_imgs, train=False)
            
            self._calculate_metrics(ref_imgs, pred_imgs, train=False)

            full_img = self._gen_comparison_img(inp_imgs, pred_imgs, ref_imgs)
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)
    
    def test(self):
        for epoch in self.epochs:
            self.test_one_epoch(epoch, self.test_name, self.load_prefix)
    
    def test_one_epoch(self, epoch: int, test_name: str, load_prefix=None):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"

        if not self.logger is None:
            self.logger.info(f"Starting Test Process...")
            self.logger.info(f"model_name: {self.model_name}")
            self.logger.info(f"mode: {self.mode}")
            self.logger.info(f"device: {self.device}")
            self.logger.info(f"checkpoint_dir: {self.checkpoint_dir}")
            self.logger.info(f"net_cfg: {self.cfg['net_cfg']}")
            for k, v in self.net_cfg.items():
                self.logger.info(f"  {k}: {v}")

        
        weights_name = f"{load_prefix}_{epoch}" if load_prefix else f"{epoch}"
        self.load_network_state(f"{weights_name}.pth")
        result_dir = os.path.join(self.result_dir, test_name, weights_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(os.path.join(result_dir, 'paired'))
        os.makedirs(os.path.join(result_dir, 'single/input'))
        os.makedirs(os.path.join(result_dir, 'single/predicted'))

        t_elapse_list = []
        idx = 1
        for batch in tqdm(self.test_dl):
            inp_imgs = batch['inp'].to(self.device)
            ref_imgs = batch['ref'].to(self.device) if 'ref' in batch else None
            img_names = batch['img_name']
            num = len(inp_imgs)
            with torch.no_grad():
                self.network.eval()
                t_start = time.time()
                pred_imgs = self.network(inp_imgs)
                
                # average inference time consumed by one batch
                t_elapse_avg = (time.time() - t_start) / num
                t_elapse_list.append(t_elapse_avg)

                # record visual results and metrics values
                full_img = self._gen_comparison_img(inp_imgs, pred_imgs, ref_imgs)
                full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(result_dir, 'paired', f'{idx:06d}.png'), full_img)
                with open(os.path.join(result_dir, 'paired', f"{idx:06d}.txt"), 'w') as f:
                    f.write('\n'.join(img_names))
                if not ref_imgs is None:
                    for (img_name, inp_img, pred_img, ref_img) in zip(
                        img_names, inp_imgs, pred_imgs, ref_imgs):
                        save_image(inp_img.data,
                                os.path.join(result_dir, 'single/input', img_name))
                        save_image(pred_img.data,
                                os.path.join(result_dir, 'single/predicted', img_name))
                else:
                    for (img_name, inp_img, pred_img) in zip(
                        img_names, inp_imgs, pred_imgs):
                        save_image(inp_img.data,
                                os.path.join(result_dir, 'single/input', img_name))
                        save_image(pred_img.data,
                                os.path.join(result_dir, 'single/predicted', img_name))
            idx += 1

        frame_rate = 1 / (sum(t_elapse_list) / len(t_elapse_list))
        if self.logger:
            self.logger.info(
                '[epoch: {:d}] [framte_rate: {:.1f} fps]'.format(
                    epoch, frame_rate
                )
            )
    
    def _gen_comparison_img(self, inp_imgs: Tensor, pred_imgs: Tensor, ref_imgs: Union[Tensor, None]=None):
        inp_imgs = torch.cat([t for t in inp_imgs], dim=2)
        pred_imgs = torch.cat([t for t in pred_imgs], dim=2)
        inp_imgs = (inp_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        pred_imgs = (pred_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        if not ref_imgs is None:
            ref_imgs = torch.cat([t for t in ref_imgs], dim=2)
            ref_imgs = (ref_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)      
            full_img = np.concatenate((inp_imgs, pred_imgs, ref_imgs), axis=0)
        else:
            full_img = np.concatenate((inp_imgs, pred_imgs), axis=0)

        return full_img
    
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument('--lambda_mae', type=float, default=1.0, help='weight of MAE loss')
            parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight of SSIM loss')
            parser.add_argument('--lambda_psnr', type=float, default=1.0, help='weight of PSNR loss')
            parser.add_argument('--lambda_four', type=float, default=1.0, help='weight of FourDomain loss')
            parser.add_argument('--lambda_edge', type=float, default=1.0, help='weight of Edge loss')
            parser.add_argument('--l1_reduction', type=str, default='mean')
        return parser


@_models.register('ie2')
class ImgEnhanceModel2(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.l1charbonnier_loss_fn = L1CharbonnierLoss().to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.lambda_l1charbonnier  = self.cfg['lambda_l1charbonnier']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['l1charbonnier'] = self.l1charbonnier_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_l1charbonnier * loss['l1charbonnier'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr']
        
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_l1charbonnier", type=float, default=1.0, help="weight of L1 Charbonnier loss")
            parser.add_argument('--lambda_ssim', type=float, default=1.0, help='weight of SSIM loss')
            parser.add_argument('--lambda_psnr', type=float, default=1.0, help='weight of PSNR loss')
        return parser


@_models.register('ie3')
class ImgEnhanceModel3(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.four_loss_fn = FourDomainLoss().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
        self.lambda_four = self.cfg['lambda_four']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['four'] = self.four_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr'] +\
            self.lambda_four * loss['four']
        
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--lambda_psnr", type=float, default=1.0, help="weight of PSNR loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser


@_models.register('ie4')
class ImgEnhanceModel4(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.four_loss_fn = FourDomainLoss3().to(self.device)
        self.edge_loss_fn = EdgeLoss().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
        self.lambda_four = self.cfg['lambda_four']
        self.lambda_edge = self.cfg['lambda_edge']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['four'] = self.four_loss_fn(pred_imgs, ref_imgs)
        loss['edge'] = self.edge_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr'] +\
            self.lambda_four * loss['four'] +\
            self.lambda_edge * loss['edge']

    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--lambda_psnr", type=float, default=1.0, help="weight of PSNR loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--lambda_edge", type=float, default=1.0, help="weight of Edge loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser


@_models.register('ie5')
class ImgEnhanceModel5(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.four_loss_fn = FourDomainLoss2().to(self.device)
        self.edge_loss_fn = EdgeLoss().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
        self.lambda_four = self.cfg['lambda_four']
        self.lambda_edge = self.cfg['lambda_edge']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['four'] = self.four_loss_fn(pred_imgs, ref_imgs)
        loss['edge'] = self.edge_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr'] +\
            self.lambda_four * loss['four'] +\
            self.lambda_edge * loss['edge']
    
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--lambda_psnr", type=float, default=1.0, help="weight of PSNR loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--lambda_edge", type=float, default=1.0, help="weight of Edge loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser
        

@_models.register('ie6')
class ImgEnhanceModel6(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.s3im_loss_fn = S3IM(
            repeat_time=self.cfg['repeat_time'],
            patch_height=self.cfg['patch_height'],
            patch_width=self.cfg['patch_width']
        ).to(self.device).to(self.device)
        self.four_loss_fn = FourDomainLoss2().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_s3im = self.cfg['lambda_s3im']
        self.lambda_four = self.cfg['lambda_four']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        n = len(pred_imgs)
        loss['s3im'] = self.s3im_loss_fn(pred_imgs.view(n, -1), ref_imgs.view(n, -1))
        loss['four'] = self.four_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_s3im * loss['s3im'] +\
            self.lambda_four * loss['four']
        
    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_s3im", type=float, default=1.0, help="weight of S3IM loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--repeat_time", type=int, default=10, help="repeat time for calculating S3IM Loss")
            parser.add_argument("--patch_height", type=int, default=512, help="patch height for calculating S3IM Loss")
            parser.add_argument("--patch_width", type=int, default=512, help="patch width for calculating S3IM Loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser
        

@_models.register('ie7')
class ImgEnhanceModel7(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.four_loss_fn = FourDomainLoss2().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_four = self.cfg['lambda_four']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        ref_imgs_dict = dict()
        scales = list(pred_imgs.keys())
        for scale in scales:
            if scale == 1:
                ref_imgs_dict[1] = ref_imgs
            else:
                ref_imgs_dict[scale] = F.interpolate(ref_imgs, scale_factor=scale, mode='bilinear')
        loss['mae'] = sum(self.mae_loss_fn(pred_imgs[scale], ref_imgs_dict[scale]) for scale in scales)
        loss['ssim'] = sum(self.ssim_loss_fn(pred_imgs[scale], ref_imgs_dict[scale]) for scale in scales)
        loss['four'] = sum(self.four_loss_fn(pred_imgs[scale], ref_imgs_dict[scale]) for scale in scales)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_four * loss['four']
        
    def _calculate_metrics(self, ref_imgs, pred_imgs, train=True):
        metrics = self.train_metrics if train else self.val_metrics
        metrics['psnr'] = psnr(pred_imgs[1], ref_imgs, 1.0)
        metrics['ssim'] = ssim(pred_imgs[1], ref_imgs, 11).mean()
        
    def _gen_comparison_img(self, inp_imgs: Tensor, pred_imgs: Dict[float, Tensor], ref_imgs: Union[Tensor, None]=None):
        inp_imgs = torch.cat([t for t in inp_imgs], dim=2)
        pred_imgs = torch.cat([t for t in pred_imgs[1]], dim=2)
        inp_imgs = (inp_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        pred_imgs = (pred_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        if not ref_imgs is None:
            ref_imgs = torch.cat([t for t in ref_imgs], dim=2)
            ref_imgs = (ref_imgs.cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)      
            full_img = np.concatenate((inp_imgs, pred_imgs, ref_imgs), axis=0)
        else:
            full_img = np.concatenate((inp_imgs, pred_imgs), axis=0)

        return full_img
    
    def test(self, test_dl: DataLoader, epoch: int, test_name: str, load_prefix=None):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"
        
        weights_name = f"{load_prefix}_{epoch}" if load_prefix else f"{epoch}"
        self.load_network_state(f"{weights_name}.pth")
        result_dir = os.path.join(self.result_dir, test_name, weights_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)
        os.makedirs(os.path.join(result_dir, 'paired'))
        os.makedirs(os.path.join(result_dir, 'single/input'))
        os.makedirs(os.path.join(result_dir, 'single/predicted'))

        t_elapse_list = []
        idx = 1
        for batch in tqdm(test_dl):
            inp_imgs = batch['inp'].to(self.device)
            ref_imgs = batch['ref'].to(self.device) if 'ref' in batch else None
            img_names = batch['img_name']
            num = len(inp_imgs)
            with torch.no_grad():
                self.network.eval()
                t_start = time.time()
                pred_imgs = self.network(inp_imgs)
                
                # average inference time consumed by one batch
                t_elapse_avg = (time.time() - t_start) / num
                t_elapse_list.append(t_elapse_avg)

                # record visual results and metrics values
                full_img = self._gen_comparison_img(inp_imgs, pred_imgs, ref_imgs)
                full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(result_dir, 'paired', f'{idx:06d}.png'), full_img)
                with open(os.path.join(result_dir, 'paired', f"{idx:06d}.txt"), 'w') as f:
                    f.write('\n'.join(img_names))
                if not ref_imgs is None:
                    for (img_name, inp_img, pred_img, ref_img) in zip(
                        img_names, inp_imgs, pred_imgs[1], ref_imgs):
                        save_image(inp_img.data,
                                os.path.join(result_dir, 'single/input', img_name))
                        save_image(pred_img.data,
                                os.path.join(result_dir, 'single/predicted', img_name))
                else:
                    for (img_name, inp_img, pred_img) in zip(
                        img_names, inp_imgs, pred_imgs[1]):
                        save_image(inp_img.data,
                                os.path.join(result_dir, 'single/input', img_name))
                        save_image(pred_img.data,
                                os.path.join(result_dir, 'single/predicted', img_name))
            idx += 1

        frame_rate = 1 / (sum(t_elapse_list) / len(t_elapse_list))
        if self.logger:
            self.logger.info(
                '[epoch: {:d}] [framte_rate: {:.1f} fps]'.format(
                    epoch, frame_rate
                )
            )

    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser


@_models.register('ie8')
class ImgEnhanceModel8(ImgEnhanceModel7):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.four_loss_fn = FourDomainLoss3().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_four = self.cfg['lambda_four']

    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--lambda_four", type=float, default=1.0, help="weight of FourDomain loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser


@_models.register('aqmamba')
class AquaticMamba(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_win_size = self.cfg.get('ssim_win_size', 11)
        self.ssim_loss_fn = SSIMLoss(self.ssim_win_size).to(self.device)
        self.use_contrast_loss = self.cfg.get('use_constrast_loss', False)
        if self.use_contrast_loss:
            self.contrast_loss_fn = ContrastLoss().to(self.device)
        self.lambda_mae  = self.cfg.get('lambda_mae', 1.0)
        self.lambda_ssim = self.cfg.get('lambda_ssim', 1.0)

    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + self.lambda_ssim * loss['ssim']
        if self.use_contrast_loss:
            loss['contrast'] = self.contrast_loss_fn(pred_imgs, ref_imgs)
            loss['total'] += loss['contrast']

    def _set_optimizer(self):
        params = [{'params': self.network.parameters()}, ]
        if self.use_contrast_loss:
            params.append({'params': self.contrast_loss_fn.parameters()})
        if len(params) == 1:
            params = self.network.parameters()
        optimizer = self.cfg['optimizer']
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=self.cfg['lr'])
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=self.cfg['lr'])
        else:
            assert f"<{optimizer}> is supported!"

    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument("--use_contrast_loss", action='store_true', help="whether use contrast loss function")
            parser.add_argument("--lambda_mae", type=float, default=1.0, help="weight of MAE loss")
            parser.add_argument("--lambda_ssim", type=float, default=1.0, help="weight of SSIM loss")
            parser.add_argument("--l1_reduction", type=str, default='mean')
        return parser