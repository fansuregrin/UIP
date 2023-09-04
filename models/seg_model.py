import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import os
import numpy as np
import time
import cv2
from torchvision.utils import draw_segmentation_masks
from typing import Dict

from .base_model import BaseModel
from .loss import DiceLoss


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