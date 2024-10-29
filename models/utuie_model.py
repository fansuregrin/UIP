import os
import random
import sys
import shutil
import yaml
import time
from typing import Dict, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import cv2
import numpy as np
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from kornia.metrics import psnr, ssim
from kornia.losses import SSIMLoss

from .base_model import BaseModel
from losses import LABLoss, LCHLoss, VGG19_PercepLoss
from models import _models
from networks import create_network
from data import create_dataloader, create_dataset
from utils import seed_everything


@_models.register('utuie')
class UTUIE(BaseModel):
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
            os.makedirs(os.path.join(self.network_state_dir, 'G'), exist_ok=True)
            os.makedirs(os.path.join(self.network_state_dir, 'D'), exist_ok=True)
            # Set optimizers
            self._set_optimizer()
            self.optimizer_state_dir = os.path.join(self.checkpoint_dir, 'optimizer')
            os.makedirs(os.path.join(self.optimizer_state_dir, 'G'), exist_ok=True)
            os.makedirs(os.path.join(self.optimizer_state_dir, 'D'), exist_ok=True)
            # Set lr_scheduler
            self._set_lr_scheduler()
            self.lr_scheduler_state_dir = os.path.join(self.checkpoint_dir, 'lr_scheduler')
            os.makedirs(os.path.join(self.lr_scheduler_state_dir, 'G'), exist_ok=True)
            os.makedirs(os.path.join(self.lr_scheduler_state_dir, 'D'), exist_ok=True)
            # Set Loss function
            self._set_loss_fn()
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

    def split(self, img: torch.Tensor):
        output=[]
        output.append(F.interpolate(img, scale_factor=0.125))
        output.append(F.interpolate(img, scale_factor=0.25))
        output.append(F.interpolate(img, scale_factor=0.5))
        output.append(img)
        for i in range(len(output)):
            output[i] = output[i].to(self.device)
        return output
    
    def _set_loss_fn(self):
        self.criterion_GAN = nn.MSELoss(reduction='sum').to(self.device)
        self.criterion_pixelwise = nn.MSELoss(reduction='sum').to(self.device)
        self.mse_loss_fn  = nn.MSELoss(reduction='sum').to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.vgg_loss_fn = VGG19_PercepLoss().to(self.device)
        self.lab_loss_fn = LABLoss().to(self.device)
        self.lch_loss_fn = LCHLoss().to(self.device)
        self.lambda_pixel = self.cfg['lambda_pixel']
        self.lambda_lab = self.cfg['lambda_lab']
        self.lambda_lch = self.cfg['lambda_lch']
        self.lambda_con = self.cfg['lambda_con']
        self.lambda_ssim = self.cfg['lambda_ssim']

    def _set_optimizer(self):
        optimizer = self.cfg['optimizer']
        self.optimizer = {}
        if optimizer == 'adam':
            self.optimizer['G'] = torch.optim.Adam(
                self.network['G'].parameters(), lr=self.cfg['lr'])
            self.optimizer['D'] = torch.optim.Adam(
                self.network['D'].parameters(), lr=self.cfg['lr'])
        elif optimizer == 'sgd':
            self.optimizer['G'] = torch.optim.SGD(
                self.network['G'].parameters(), lr=self.cfg['lr'])
            self.optimizer['D'] = torch.optim.SGD(
                self.network['D'].parameters(), lr=self.cfg['lr'])
        else:
            assert f"<{optimizer}> is supported!"

    def _set_lr_scheduler(self):
        with open(self.cfg['lr_scheduler_cfg']) as f:
            lr_scheduler_cfg = yaml.load(f, yaml.FullLoader)
        self.lr_scheduler = {}
        if lr_scheduler_cfg['name'] == 'none':
            self.lr_scheduler['G'] = None
            self.lr_scheduler['D'] = None
        elif lr_scheduler_cfg['name'] == 'step_lr':
            self.lr_scheduler['G'] = optim.lr_scheduler.StepLR(
                self.optimizer['G'], lr_scheduler_cfg['step_size'],
                lr_scheduler_cfg['gamma'])
            self.lr_scheduler['D'] = optim.lr_scheduler.StepLR(
                self.optimizer['D'], lr_scheduler_cfg['step_size'],
                lr_scheduler_cfg['gamma'])
        else:
            assert f"<{lr_scheduler_cfg['name']}> is supported!"

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

    def load_network_state(self, state_name: str, label: str):
        state_path = os.path.join(self.network_state_dir, label, state_name)
        self.network[label].load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info("Loaded network['{}'] weights from {}.".format(
                label, state_path
            ))

    def load_optimizer_state(self, state_name: str, label: str):
        state_path = os.path.join(self.optimizer_state_dir, label, state_name)
        self.optimizer[label].load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info("Loaded optimizer['{}'] state from {}.".format(
                label, state_path
            ))

    def load_lr_scheduler_state(self, state_name: str, label: str):
        state_path = os.path.join(self.lr_scheduler_state_dir, label, state_name)
        self.lr_scheduler[label].load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info("Loaded lr_scheduler['{}'] state from {}.".format(
                label, state_path
            ))
    
    def _calculate_metrics(self, ref_imgs, pred_imgs, train=True):
        metrics = self.train_metrics if train else self.val_metrics
        metrics['psnr'] = psnr(pred_imgs, ref_imgs, 1.0)
        metrics['ssim'] = ssim(pred_imgs, ref_imgs, 11).mean()

    def train(self):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"

        seed_everything(self.seed)

        if not self.logger is None:
            self.logger.info(f"Starting Training Process...")
            self.logger.info(f"model_name: {self.model_name}")
            self.logger.info(f"mode: {self.mode}")
            self.logger.info(f"device: {self.device}")
            self.logger.info(f"checkpoint_dir: {self.checkpoint_dir}")
            self.logger.info(f"net_cfg: {self.cfg['net_cfg']}")
            for k, v in self.net_cfg.items():
                self.logger.info(f"  {k}: {v}")

        if self.start_epoch > 0:
            load_prefix = self.cfg.get('load_prefix', None)
            if load_prefix:
                state_name = f'{load_prefix}_{self.start_epoch-1}.pth'
                for label in ('G', 'D'):
                    self.load_network_state(state_name, label)
                    self.load_optimizer_state(state_name, label)
                    self.load_lr_scheduler_state(state_name, label)
            else:
                state_name = f'{self.start_epoch-1}.pth'
                for label in ('G', 'D'):
                    self.load_network_state(state_name, label)
                    self.load_optimizer_state(state_name, label)
                    self.load_lr_scheduler_state(state_name, label)
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

                    self.logger.info(
                        "[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] "
                        "[train_loss: G:{:.3f}, D:{:.3f}, val_loss: G:{:.3f}, D:{:.3f}]".format(
                            iteration_index, self.optimizer['G'].param_groups[0]['lr'],
                            epoch, self.start_epoch + self.num_epochs-1, i, len(self.train_dl)-1,
                            self.train_loss['loss_G'].item(), self.train_loss['loss_D'].item(),
                            self.val_loss['loss_G'].item(), self.val_loss['loss_D'].item()
                    ))
                iteration_index += 1
            # adjust lr
            self.adjust_lr()
            # save model weights
            if (epoch % self.ckpt_interval == 0) or (epoch == self.start_epoch + self.num_epochs-1):
                for label in ('G', 'D'):
                    self.save_network_weights(epoch, label)
                    self.save_optimizer_state(epoch, label)
                    self.save_lr_scheduler_state(epoch, label)

    def train_one_batch(self, input_: Dict):
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        inp_split = self.split(inp_imgs)
        ref_split = self.split(ref_imgs)
        split_len = len(inp_split)

        patch = (1, 256 // 2 ** 5, 256// 2 ** 5)
        valid = torch.ones((inp_imgs.size(0), *patch), requires_grad=True, device=self.device)
        fake = torch.zeros((inp_imgs.size(0), *patch), requires_grad=True, device=self.device)
        
        ## Train Generator
        self.optimizer['G'].zero_grad()
        self.network['G'].train()
        pred_imgs = self.network['G'](inp_imgs)
        d_pred = self.network['D'](pred_imgs, inp_split)

        loss_GAN = self.criterion_GAN(d_pred, valid)
        loss_pixel = sum(self.criterion_pixelwise(pred_imgs[0], ref_split[0]) for i in range(split_len)) / split_len
        loss_ssim = -sum(self.ssim_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
        loss_con = sum(self.vgg_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
        loss_lab = sum(self.lab_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
        loss_lch = sum(self.lch_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
        loss_G = loss_GAN + self.lambda_pixel * loss_pixel + \
            self.lambda_ssim * loss_ssim + self.lambda_con * loss_con + \
            self.lambda_lab * loss_lab + self.lambda_lch * loss_lch
        loss_G.backward()
        self.optimizer['G'].step()

        ## Train Discriminator
        self.optimizer['D'].zero_grad()
        self.network['D'].train()
        d_real = self.network['D'](ref_split, inp_split)
        for i in range(len(pred_imgs)):
            pred_imgs[i] = pred_imgs[i].detach()
        d_fake = self.network['D'](pred_imgs, inp_split)
        
        loss_real = self.criterion_GAN(d_real, valid)
        loss_fake = self.criterion_GAN(d_fake, fake)
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        self.optimizer['D'].step()

        self.train_loss['loss_G'] = loss_G
        self.train_loss['loss_D'] = loss_D
        self._calculate_metrics(ref_imgs, pred_imgs[-1])
    
    def adjust_lr(self):
        for label in self.lr_scheduler:
            if self.lr_scheduler[label] is not None:
                self.lr_scheduler[label].step()
    
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
    
    def save_network_weights(self, epoch: int, label: str):
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.network_state_dir, label, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.network_state_dir, label, "{:d}.pth".format(epoch))
        torch.save(self.network[label].state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved network['{}'] weights into {}".format(label, saved_path))

    def save_optimizer_state(self, epoch: int, label: str):
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.optimizer_state_dir, label, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.optimizer_state_dir, label, "{:d}.pth".format(epoch))
        torch.save(self.optimizer[label].state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved optimizer['{}'] state into {}".format(label, saved_path))

    def save_lr_scheduler_state(self, epoch: int, label: str):
        if self.lr_scheduler[label] is None:
            return
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.lr_scheduler_state_dir, label, "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.lr_scheduler_state_dir, label, "{:d}.pth".format(epoch))
        torch.save(self.lr_scheduler[label].state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved lr_shceduler['{}'] state into {}".format(label, saved_path))

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        inp_split = self.split(inp_imgs)
        ref_split = self.split(ref_imgs)
        split_len = len(inp_split)
        patch = (1, 256 // 2 ** 5, 256// 2 ** 5)
        valid = torch.ones((inp_imgs.size(0), *patch), requires_grad=False, device=self.device)
        fake = torch.zeros((inp_imgs.size(0), *patch), requires_grad=False, device=self.device)
        with torch.no_grad():
            self.network['G'].eval()
            self.network['D'].eval()
            pred_imgs = self.network['G'](inp_imgs)
            d_pred = self.network['D'](pred_imgs, inp_split)

            loss_GAN = self.criterion_GAN(d_pred, valid)
            loss_pixel = sum(self.criterion_pixelwise(pred_imgs[0], ref_split[0]) for i in range(split_len)) / split_len
            loss_ssim = -sum(self.ssim_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
            loss_con = sum(self.vgg_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
            loss_lab = sum(self.lab_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
            loss_lch = sum(self.lch_loss_fn(pred_imgs[i], ref_split[i]) for i in range(split_len)) / split_len
            loss_G = loss_GAN + self.lambda_pixel * loss_pixel + \
                self.lambda_ssim * loss_ssim + self.lambda_con * loss_con + \
                self.lambda_lab * loss_lab + self.lambda_lch * loss_lch
            
            d_real = self.network['D'](ref_split, inp_split)
            for i in range(len(pred_imgs)):
                pred_imgs[i] = pred_imgs[i].detach()
            d_fake = self.network['D'](pred_imgs, inp_split)
            
            loss_real = self.criterion_GAN(d_real, valid)
            loss_fake = self.criterion_GAN(d_fake, fake)
            loss_D = 0.5 * (loss_real + loss_fake)

            self.val_loss['loss_G'] = loss_G
            self.val_loss['loss_D'] = loss_D

            self._calculate_metrics(ref_imgs, pred_imgs[-1], train=False)

            full_img = self._gen_comparison_img(inp_imgs, pred_imgs[-1], ref_imgs)
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)
    
    def test(self):
        for epoch in self.epochs:
            self.test_one_epoch(epoch, self.test_name, self.load_prefix)

    def test_one_epoch(self, epoch: int, test_name: str, load_prefix=None):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"
        
        weights_name = f"{load_prefix}_{epoch}" if load_prefix else f"{epoch}"
        self.load_network_state(f"{weights_name}.pth", 'G')
        result_dir = os.path.join(self.result_dir, test_name, weights_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)
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
                self.network['G'].eval()
                t_start = time.time()
                pred_imgs = self.network['G'](inp_imgs)[-1]
                
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
            parser.add_argument('--lambda_pixel', type=float, default=0.1)
            parser.add_argument('--lambda_lab', type=float, default=0.001)
            parser.add_argument('--lambda_lch', type=float, default=1)
            parser.add_argument('--lambda_con', type=float, default=100)
            parser.add_argument('--lambda_ssim', type=float, default=100)
        return parser