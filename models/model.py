import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import time
from torch import Tensor
from abc import ABC
from abc import abstractmethod
from typing import Union, Dict, Any
from kornia.losses import SSIMLoss, PSNRLoss
from kornia.metrics import psnr, ssim
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import cv2
import shutil

from .initialize import init_weight_bias
from .networks import create_network
from .loss import DiceLoss


def create_model(name, cfg):
    if name == 'ie':
        model = ImgEnhanceModel(cfg)
    if name == 'seg':
        model = SegModel(cfg)
    else:
        assert f"<{name}> not exist!"
    return model

class BaseModel(ABC):
    """Abstract base class (ABC) for models.
    """
    def __init__(self, cfg: Dict[str, Any]):
        """Initialize the BaseModel class.

        Args:
            cfg: Configurations, a `Dict`.
        """
        self.device = cfg.get('device', torch.device('cpu'))
        self.logger =  cfg.get('logger', None)
        self.net_cfg = cfg['net_cfg']
        self.mode = cfg['mode']
        self.cfg = cfg
        self.setup()

    def setup(self):
        """Setup the model.
        """
        self.network = create_network(self.net_cfg).to(self.device)
        if self.mode == 'train':
            self.tb_writer = self.cfg['tb_writer']
            self.sample_dir = self.cfg['sample_dir']
            self.checkpoint_dir = self.cfg['checkpoint_dir']
            self.name = self.cfg['name']
            self.start_epoch = self.cfg['start_epoch']
            self.start_iteration = self.cfg['start_iteration']
            self.num_epochs = self.cfg['num_epochs']
            self.val_interval = self.cfg['val_interval']
            self.ckpt_interval = self.cfg['ckpt_interval']
            if self.start_epoch == 0:
                self.network.apply(init_weight_bias)
        else:
            assert f"{self.mode} is not supported!"


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
            self.lambda_mae = cfg['lambda_mae']
            self.lambda_ssim = cfg['lambda_ssim']
            self.lambda_psnr = cfg['lambda_psnr']
            self.train_loss = {}
            self.val_loss = {}
            self.train_metrics = {}
            self.val_metrics = {}
        elif self.mode == 'test':
            self.checkpoint_dir = cfg['checkpoint_dir']
            self.result_dir = cfg['result_dir']
        
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
        self.mae_loss_fn = nn.L1Loss().to(self.device)
        self.ssim_loss_fn = SSIMLoss(11).to(self.device)
        self.psnr_loss_fn = PSNRLoss(1.0).to(self.device)

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
        self.train_loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        self.train_loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
        self.train_loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
        self.train_loss['total'] = self.lambda_mae * self.train_loss['mae'] + \
                          self.lambda_ssim * self.train_loss['ssim'] +\
                          self.lambda_psnr * self.train_loss['psnr']
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
            
            self.val_loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
            self.val_loss['ssim'] = self.ssim_loss_fn(pred_imgs, ref_imgs)
            self.val_loss['psnr'] = self.psnr_loss_fn(pred_imgs, ref_imgs)
            self.val_loss['total'] = self.lambda_mae * self.val_loss['mae'] + \
                            self.lambda_ssim * self.val_loss['ssim'] +\
                            self.lambda_psnr * self.val_loss['psnr']
            
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
        psnr_val_list = []
        ssim_val_list = []
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
                ssim_val = ssim(pred_imgs, ref_imgs, 11, 1.0).mean().item()
                ssim_val_list.append(ssim_val)
                for (img_name, inp_img, pred_img, ref_img) in zip(
                    img_names, inp_imgs, pred_imgs, ref_imgs):
                    psnr_val = psnr(pred_img, ref_img, 1.0).item()
                    psnr_val_list.append(psnr_val)
                    save_image(inp_img.data,
                               os.path.join(result_dir, 'single/input', img_name))
                    save_image(pred_img.data,
                               os.path.join(result_dir, 'single/predicted', img_name))
            idx += 1

        frame_rate = 1 / (sum(t_elapse_list) / len(t_elapse_list))
        psnr_val = sum(psnr_val_list) / len(psnr_val_list)
        ssim_val = sum(ssim_val_list) / len(ssim_val_list)

        with open(
            os.path.join(result_dir, 'metrics.csv'),
            'w') as f:
            f.write('frame_rate,psnr,ssim\n'
                    '{:.1f},{:.3f},{:.3f}'.format(frame_rate, psnr_val, ssim_val))
        if self.logger:
            self.logger.info(
                '[epoch: {:d}] [framte_rate: {:.1f} fps, psnr: {:.3f}, ssim: {:.3f}]'.format(
                    epoch, frame_rate, psnr_val, ssim_val
                )
            )

        return (frame_rate, psnr_val, ssim_val)
    
    def _gen_comparison_img(self, inp_imgs: Tensor, pred_imgs: Tensor, ref_imgs: Union[Tensor, None]):
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
    

class SegModel(BaseModel):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.color_map = cfg['color_map']
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
            self.lambda_ce = cfg['lambda_ce']
            self.lambda_dice = cfg['lambda_dice']

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
        self.ce_loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.dice_loss_fn = DiceLoss('micro').to(self.device)

    def train(self, input_: Dict):
        inp_imgs = input_['img'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        self.optimizer.zero_grad()
        self.network.train()
        pred_masks = self.network(inp_imgs)
        self.train_metrics['mIoU'] = self.calc_mIOU(pred_masks, ref_masks, len(self.color_map))
        self.train_loss['ce'] = self.ce_loss_fn(pred_masks, ref_masks)
        self.train_loss['dice'] = self.dice_loss_fn(pred_masks, ref_masks)
        self.train_loss['total'] = self.train_loss['ce'] * self.lambda_ce +\
                                   self.train_loss['dice'] * self.lambda_dice
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
    
    def save_model_weights(self, epoch):
        saved_path = os.path.join(self.checkpoint_dir, "weights_{:d}.pth".format(epoch))
        torch.save(self.network.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved model weights into {}".format(saved_path))

    def validate(self, input_: Dict, iteration):
        inp_imgs = input_['img'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        with torch.no_grad():
            pred_masks = self.network(inp_imgs)

            self.val_loss['ce'] = self.ce_loss_fn(pred_masks, ref_masks)
            self.val_loss['dice'] = self.dice_loss_fn(pred_masks, ref_masks)
            self.val_loss['total'] = self.val_loss['ce'] * self.lambda_ce +\
                                     self.val_loss['dice'] * self.lambda_dice
            self.val_metrics['mIoU'] = self.calc_mIOU(pred_masks, ref_masks, len(self.color_map))
            
            full_img = self._gen_comparison_img(inp_imgs.cpu(), pred_masks.cpu(), ref_masks.cpu())
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)
    
    def test(self, input_: Dict):
        inp_imgs = input_['img'].to(self.device)
        ref_masks = input_['mask'].to(self.device)
        num = len(inp_imgs)
        with torch.no_grad():
            self.network.eval()
            t_start = time.time()
            pred_masks = self.network(inp_imgs)
            t_elapse_avg = (time.time() - t_start) / num
        full_img = self._gen_comparison_img(inp_imgs.cpu(), pred_masks.cpu(), ref_masks.cpu())

        return full_img, t_elapse_avg
    
    def _gen_comparison_img(self, imgs: Tensor, pred_masks: Tensor, ref_masks):
        colors = ['#'+c for c in sorted(self.color_map.keys())]
        imgs_with_pred_mask = []
        imgs_with_ref_mask = []
        for img, pred_mask, ref_mask in zip(imgs, pred_masks, ref_masks):
            img = (img * 255).to(torch.uint8)
            normalized_pred_mask = F.softmax(pred_mask, dim=0)
            boolean_pred_mask = torch.stack(
                [(normalized_pred_mask.argmax(0) == i) for i in range(len(self.color_map))]
            )
            img_with_pred_mask = draw_segmentation_masks(img, boolean_pred_mask, alpha=0.5, colors=colors)
            imgs_with_pred_mask.append(img_with_pred_mask.cpu().numpy().transpose(1,2,0))

            normalized_ref_mask = F.softmax(ref_mask, dim=0)
            boolean_ref_mask = torch.stack(
                [(normalized_ref_mask.argmax(0) == i) for i in range(len(self.color_map))]
            )
            img_with_ref_mask = draw_segmentation_masks(img, boolean_ref_mask, alpha=0.5, colors=colors)
            imgs_with_ref_mask.append(img_with_ref_mask.cpu().numpy().transpose(1,2,0))
        imgs_with_pred_mask = np.concatenate(imgs_with_pred_mask, axis=1)
        imgs_with_ref_mask = np.concatenate(imgs_with_ref_mask, axis=1)
        full_img = np.concatenate((imgs_with_pred_mask, imgs_with_ref_mask), axis=0)

        return full_img
    
    def calc_mIOU(self, pred, label, num_classes):
        pred = F.softmax(pred, dim=1)
        pred = pred.argmax(1).squeeze(1)
        label = label.argmax(1).squeeze(1)
        iou_list = list()
        present_iou_list = list()

        pred = pred.view(-1)
        label = label.view(-1)
        for sem_class in range(num_classes):
            pred_inds = (pred == sem_class)
            target_inds = (label == sem_class)
            if target_inds.long().sum().item() == 0:
                iou_now = float('nan')
            else: 
                intersection_now = (pred_inds[target_inds]).long().sum().item()
                union_now = pred_inds.long().sum().item() +\
                      target_inds.long().sum().item() - intersection_now
                iou_now = float(intersection_now) / float(union_now)
                present_iou_list.append(iou_now)
            iou_list.append(iou_now)
        return np.mean(present_iou_list)