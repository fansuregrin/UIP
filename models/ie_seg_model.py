import os
import time
import sys
import random
import shutil
from argparse import ArgumentParser

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
from torchvision.utils import draw_segmentation_masks
from kornia.losses import DiceLoss, SSIMLoss
from kornia.metrics import psnr, ssim
from tqdm import tqdm
import cv2


from networks import create_network
from data import create_dataset, create_dataloader
from utils import seed_everything
from .base_model import BaseModel
from metrics import StreamSegMetrics
from models import _models


@_models.register('ieseg')
class IeSegModel(BaseModel):
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
        self.ie_net_state_dir = os.path.join(self.checkpoint_dir, 'ie_net')
        os.makedirs(self.ie_net_state_dir, exist_ok=True)
        self.seg_net_state_dir = os.path.join(self.checkpoint_dir, 'seg_net')
        os.makedirs(self.seg_net_state_dir, exist_ok=True)
        
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
        optim_params_cfg = self.cfg.get('optim_params_cfg', None)
        if optim_params_cfg:
            with open(optim_params_cfg) as f:
                _cfg = yaml.load(f, yaml.FullLoader)
            ie_net_params = []
            seg_net_params = []
            for name,param in self.ie_net.named_parameters():
                if name in _cfg['ie']:
                    ie_net_params.append(param)
            for name,param in self.seg_net.named_parameters():
                if name in _cfg['seg']:
                    seg_net_params.append(param)
        else:
            ie_net_params = self.ie_net.parameters()
            seg_net_params = self.seg_net.parameters()
        params = [
            {'params': ie_net_params},
            {'params': seg_net_params}]
        
        optimizer = self.cfg['optimizer']
        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                params, lr=self.cfg['lr'], betas=self.cfg['betas'],
                weight_decay=self.cfg['weight_decay'])
        elif optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                params, lr=self.cfg['lr'], momentum=self.cfg['momentum'],
                weight_decay=self.cfg['weight_decay'])
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
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.dice_loss_fn = DiceLoss().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
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
            _train_ds = self.cfg.get('train_ds', 'train')
            train_ds = create_dataset(ds_cfg[_train_ds])
            _val_ds = self.cfg.get('val_ds', 'val')
            val_ds = create_dataset(ds_cfg[_val_ds])
            self.train_dl = create_dataloader(train_ds, dl_cfg)
            self.val_dl = create_dataloader(val_ds, dl_cfg)
        elif self.mode == 'test':
            _test_ds = self.cfg.get('test_ds', 'test')
            test_ds = create_dataset(ds_cfg[_test_ds])
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
        with open(self.cfg['net_cfg']) as f:
            self.ie_net_cfg = yaml.load(f, yaml.FullLoader)
        self.ie_net_name = self.ie_net_cfg['name']
        self.ie_net = create_network(self.ie_net_cfg)
        self.ie_net.to(self.device)
        
        with open(self.cfg['seg_net_cfg']) as f:
            self.seg_net_cfg = yaml.load(f, yaml.FullLoader)
        self.seg_net_name = self.seg_net_cfg['name']
        self.seg_net = create_network(self.seg_net_cfg)
        self.seg_net.to(self.device)

        self.net_name = self.ie_net_name + '_' + self.seg_net_name

    def _load_network_state_directly(self, network, state_path: str):
        network.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded weights from [{}]'.format(
                state_path
            ))

    def load_network_state(self, state_name: str):
        state_path = os.path.join(self.ie_net_state_dir, state_name)
        self.ie_net.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded ie network weights from [{}]'.format(
                state_path
            ))

        state_path = os.path.join(self.seg_net_state_dir, state_name)
        self.seg_net.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded seg network weights from [{}]'.format(
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

    def _log_cfg_details(self):
        if self.logger is None: return

        for k, v in self.cfg.items():
            self.logger.info(f"{k}: {v}")
        self.logger.info("ie network config details:")
        for k, v in self.ie_net_cfg.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("seg network config details:")
        for k, v in self.seg_net_cfg.items():
            self.logger.info(f"  {k}: {v}")

    def train(self):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"
        
        seed_everything(self.seed)
        if not self.logger is None:
            self.logger.info(f"Starting Training Process...")
            self._log_cfg_details()
            self.logger.info("lr_scheduler config details:")
            for k, v in self.lr_scheduler_cfg.items():
                self.logger.info(f"  {k}: {v}")

        if self.cfg.get('ie_net_load_fp', None):
            self._load_network_state_directly(self.ie_net, self.cfg['ie_net_load_fp'])
        if self.cfg.get('seg_net_load_fp', None):
            self._load_network_state_directly(self.seg_net, self.cfg['seg_net_load_fp'])

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
                if (iteration_index % self.val_interval == 0):
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
        ref_imgs = input_['ref'].to(self.device)
        ref_masks = input_['mask'].to(self.device)

        self.optimizer.zero_grad()
        self.ie_net.train()
        self.seg_net.train()
        pred_imgs = self.ie_net(inp_imgs)
        pred_masks = self.seg_net(pred_imgs)

        self._calculate_loss(ref_imgs, pred_imgs, ref_masks, pred_masks)
        self._calculate_metrics(ref_imgs, pred_imgs, ref_masks, pred_masks)

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
            saved_path = os.path.join(self.ie_net_state_dir,
                "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.ie_net_state_dir, "{:d}.pth".format(epoch))
        torch.save(self.ie_net.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved ie network weights into {}".format(saved_path))

        if save_prefix:
            saved_path = os.path.join(self.seg_net_state_dir,
                "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.seg_net_state_dir, "{:d}.pth".format(epoch))
        torch.save(self.seg_net.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved seg network weights into {}".format(saved_path))

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

    def _calculate_loss(self, ref_imgs, pred_imgs, ref_masks, pred_masks, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['ce'] = self.ce_loss_fn(pred_masks, ref_masks)
        loss['dice'] = self.dice_loss_fn(
            F.softmax(pred_masks, dim=1),
            ref_masks)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            loss['ssim'] * self.lambda_ssim + \
            loss['ce'] * self.lambda_ce + \
            loss['dice'] * self.lambda_dice
        
    def _calculate_metrics(self, ref_imgs, pred_imgs, ref_masks, pred_masks, train=True):
        metrics = self.train_metrics if train else self.val_metrics
        metrics['psnr'] = psnr(pred_imgs, ref_imgs, 1.0)
        metrics['ssim'] = ssim(pred_imgs, ref_imgs, 11).mean()
        _metrics = StreamSegMetrics(len(self.classes))
        _metrics.reset()
        _metrics.update(ref_masks.cpu().numpy(), pred_masks.softmax(1).argmax(1).cpu().numpy())
        res = _metrics.get_results()
        metrics['mIoU'] = res['Mean_IoU']
        metrics['acc'] = res['Overall_Acc']

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['img'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        with torch.no_grad():
            pred_imgs = self.ie_net(inp_imgs)
            pred_masks = self.seg_net(pred_imgs)
            self._calculate_loss(ref_imgs, pred_imgs, ref_masks, pred_masks, train=False)
            self._calculate_metrics(ref_imgs, pred_imgs, ref_masks, pred_masks, train=False)
            sample_masks = self._sample_masks(pred_imgs.cpu(), pred_masks.cpu(), ref_masks.cpu())
            sample_masks = cv2.cvtColor(sample_masks, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}_masks.png'), sample_masks)
            sample_enh_imgs = self._sample_enhanced_imgs(inp_imgs.cpu(), pred_imgs.cpu(), ref_imgs.cpu())
            sample_enh_imgs = cv2.cvtColor(sample_enh_imgs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}_enh.png'), sample_enh_imgs)

    def test(self):
        if not self.logger is None:
            self.logger.info(f"Starting Test Process...")
            self._log_cfg_details()
        
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
        os.makedirs(os.path.join(result_dir, 'pred_masks'))
        os.makedirs(os.path.join(result_dir, 'pred_rgb_masks'))
        os.makedirs(os.path.join(result_dir, 'pred_images'))

        stream_metrics = StreamSegMetrics(len(self.classes))
        batch_idx = 1
        total_time = 0.0
        for batch in tqdm(self.test_dl):
            inp_imgs = batch['img'].to(self.device)
            ref_imgs = batch['ref'].to(self.device) if 'ref' in batch else None
            ref_masks = batch['mask'].to(self.device) if 'mask' in batch else None
            
            with torch.no_grad():
                self.ie_net.eval()
                t_start = time.time()
                pred_imgs = self.ie_net(inp_imgs)
                pred_masks = self.seg_net(pred_imgs)
                # consumed time
                t_elapse = (time.time() - t_start)
                total_time += (t_elapse / len(inp_imgs))
            
            if not ref_masks is None:
                stream_metrics.update(ref_masks.cpu().numpy(),
                    pred_masks.softmax(1).argmax(1).cpu().numpy())

            # save predicted masks
            img_names = batch['img_name']
            for pred_img,pred_mask,img_name in zip(pred_imgs, pred_masks, img_names):
                pred_mask = pred_mask.softmax(0).argmax(0)
                boolean_pred_mask = torch.stack(
                    [(pred_mask == i) for i in range(len(self.classes))]
                )

                img_name = os.path.splitext(img_name)[0] + '.png'
                save_image(pred_img, os.path.join(result_dir, 'pred_images', img_name))
                cv2.imwrite(os.path.join(result_dir, 'pred_masks', img_name),
                    pred_mask.detach().cpu().numpy())

                pred_img = (pred_img * 255).to(torch.uint8).detach().cpu()
                mask_alpha = self.cfg.get('mask_alpha', 0.5)
                colors = ['#'+clz['color'] for clz in self.classes]
                img_with_pred_mask = draw_segmentation_masks(pred_img, boolean_pred_mask,
                    alpha=mask_alpha, colors=colors).permute(1,2,0).cpu().numpy()
                cv2.imwrite(os.path.join(result_dir, 'pred_rgb_masks', img_name),
                    cv2.cvtColor(img_with_pred_mask, cv2.COLOR_RGB2BGR)) 
            
            masks = self._sample_masks(inp_imgs, pred_masks, ref_masks)
            masks = cv2.cvtColor(masks, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_dir, 'paired',
                f'{batch_idx:06d}_masks.png'), masks)
            enh_imgs = self._sample_enhanced_imgs(inp_imgs, pred_imgs, ref_imgs)
            enh_imgs = cv2.cvtColor(enh_imgs, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(result_dir, 'paired',
                f'{batch_idx:06d}_enh.png'), enh_imgs)

            batch_idx += 1

        stream_metrics = stream_metrics.get_results()
        stream_metrics['fps'] = len(self.test_dl) / total_time
        with open(os.path.join(result_dir, 'stream_metrics.yaml'), 'w') as f:
            yaml.dump(stream_metrics, f)

        if self.logger:
            self.logger.info(f"overall accurary: {stream_metrics['Overall_Acc']:.6f}")
            self.logger.info(f"mean IoU: {stream_metrics['Mean_IoU']:.6f}")
            self.logger.info(f'saved result into [{result_dir}]')
    
    def _sample_masks(self, imgs: Tensor, pred_masks: Tensor, ref_masks = None):
        mask_alpha = self.cfg.get('mask_alpha', 0.5)
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
                img_with_pred_mask = draw_segmentation_masks(img, boolean_pred_mask,
                    alpha=mask_alpha, colors=colors)
                imgs_with_pred_mask.append(img_with_pred_mask.cpu().numpy().transpose(1,2,0))

                boolean_ref_mask = torch.stack([ref_mask == i for i in range(len(self.classes))])
                img_with_ref_mask = draw_segmentation_masks(img, boolean_ref_mask,
                    alpha=mask_alpha, colors=colors)
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
                img_with_pred_mask = draw_segmentation_masks(img, boolean_pred_mask,
                    alpha=mask_alpha, colors=colors)
                imgs_with_pred_mask.append(img_with_pred_mask.cpu().numpy().transpose(1,2,0))
            imgs_with_pred_mask = np.concatenate(imgs_with_pred_mask, axis=1)
            full_img = img_with_pred_mask
        return full_img
    
    def _sample_enhanced_imgs(self, inp_imgs: Tensor, pred_imgs: Tensor, 
        ref_imgs: Union[Tensor, None]=None):
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
    def modify_args(parser: ArgumentParser, mode: str):
        parser.add_argument('--mask_alpha', type=float, default=0.5)
        parser.add_argument('--seg_net_cfg', type=str)
        if mode == 'train':
            parser.add_argument('--ie_net_load_fp', type=str, default='')
            parser.add_argument('--seg_net_load_fp', type=str, default='')
            parser.add_argument('--optim_params_cfg', type=str)
            parser.add_argument('--lambda_mae', type=float, default=1.0,
                help='weight of MAE loss')
            parser.add_argument('--l1_reduction', type=str, default='mean')
            parser.add_argument('--lambda_ssim', type=float, default=1.0,
                help='weight of SSIM loss')
            parser.add_argument('--lambda_ce', type=float, default=1.0)
            parser.add_argument('--lambda_dice', type=float, default=1.0)

        return parser