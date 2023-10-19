import torch
import torch.optim as optim
import torch.nn as nn
import os
import numpy as np
import time
from torch import Tensor
from typing import Union, Dict
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.metrics import psnr, ssim
from kornia.color import rgb_to_ycbcr

from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import torch
import pyiqa
import cv2
import shutil

from .base_model import BaseModel
from losses import (
    L1CharbonnierLoss, FourDomainLoss, 
    EdgeLoss, FourDomainLoss2,
    S3IM
)


class ImgEnhanceModel(BaseModel):
    """
    """
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        if self.mode == 'train':
            # Set optimizers
            self._set_optimizer()
            # Set lr_scheduler
            self._set_lr_scheduler()
            # Set Loss function
            self._set_loss_fn()
            self.train_loss = {}
            self.val_loss = {}
            self.train_metrics = {}
            self.val_metrics = {}
        elif self.mode == 'test':
            self.checkpoint_dir = cfg['checkpoint_dir']
            self.result_dir = cfg['result_dir']
            self.niqe = pyiqa.create_metric('niqe', device=self.device)
        
    def load_weights(self, weights_path: str):
        weights_path = os.path.join(self.checkpoint_dir, weights_path)
        self.network.load_state_dict(torch.load(weights_path))
        if self.logger:
            self.logger.info('Loaded model weights from {}.'.format(
                weights_path
            ))

    def _set_optimizer(self):
        params = self.network.parameters()
        optimizer = self.cfg['optimizer']
        if optimizer['name'] == 'adam':
            self.optimizer = torch.optim.Adam(params, lr=optimizer['lr'])
        elif optimizer['name'] == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=optimizer['lr'])
        else:
            assert f"<{optimizer['name']}> is supported!"

    def _set_lr_scheduler(self):
        lr_scheduler = self.cfg['lr_scheduler']
        if lr_scheduler['name'] == 'none':
            self.lr_scheduler = None
        elif lr_scheduler['name'] == 'step_lr':
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, lr_scheduler['step_size'],
                                                          lr_scheduler['gamma'])
        else:
            assert f"<{lr_scheduler['name']}> is supported!"

    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_ssim = self.cfg['lambda_ssim']
        self.lambda_psnr = self.cfg['lambda_psnr']
    
    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_ssim * loss['ssim'] +\
            self.lambda_psnr * loss['psnr']

    def train(self, train_dl: DataLoader, val_dl: DataLoader):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"
        assert len(train_dl) * self.start_epoch == self.start_iteration
        if self.start_epoch > 0:
            self.load_weights(f'weights_{self.start_epoch-1}.pth')
        iteration_index = self.start_iteration
        for epoch in range(self.start_epoch, self.start_epoch + self.num_epochs):
            for i, batch in enumerate(train_dl):
                # train one batch
                self.train_one_batch(batch)
                
                # validation
                if (iteration_index % self.val_interval == 0) or (i == len(train_dl)-1):
                    val_batch = next(iter(val_dl))
                    self.validate_one_batch(val_batch, iteration_index)
                    self.write_tensorboard(iteration_index)

                    self.logger.info(
                        "[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] "
                        "[train_loss: {:.3f}, val_loss: {:.3f}]".format(
                            iteration_index, self.optimizer.param_groups[0]['lr'],
                            epoch, self.start_epoch + self.num_epochs-1, i, len(train_dl)-1,
                            self.train_loss['total'].item(), self.val_loss['total'].item()
                    ))
                iteration_index += 1
            # adjust lr
            self.adjust_lr()
            # save model weights
            if (epoch % self.ckpt_interval == 0) or (epoch == self.start_epoch + self.num_epochs-1):
                self.save_model_weights(epoch)

    def train_one_batch(self, input_: Dict):
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        self.optimizer.zero_grad()
        self.network.train()
        pred_imgs = self.network(inp_imgs)
        self._calculate_loss(ref_imgs, pred_imgs)
        self.train_loss['total'].backward()
        self.optimizer.step()
        self.train_metrics['psnr'] = psnr(pred_imgs, ref_imgs, 1.0)
        self.train_metrics['ssim'] = ssim(pred_imgs, ref_imgs, 11).mean()
    
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
    
    def save_model_weights(self, epoch: int):
        saved_path = os.path.join(self.checkpoint_dir, "weights_{:d}.pth".format(epoch))
        torch.save(self.network.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved model weights into {}".format(saved_path))

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['inp'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        with torch.no_grad():
            pred_imgs = self.network(inp_imgs)
            
            self._calculate_loss(ref_imgs, pred_imgs, train=False)
            
            self.val_metrics['psnr'] = psnr(pred_imgs, ref_imgs, 1.0)
            self.val_metrics['ssim'] = ssim(pred_imgs, ref_imgs, 11).mean()

            full_img = self._gen_comparison_img(inp_imgs, pred_imgs, ref_imgs)
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)
    
    def test(self, test_dl: DataLoader, epoch: int, test_name: str):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"
        
        result_dir = os.path.join(self.result_dir, test_name, f"epoch_{epoch}")
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


class ImgEnhanceModel4(ImgEnhanceModel):
    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)
        self.four_loss_fn = FourDomainLoss().to(self.device)
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