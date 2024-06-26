import torch
import torch.optim as optim
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import numpy as np
import time
import cv2
import shutil
from tqdm import tqdm
from torchvision.utils import draw_segmentation_masks
from kornia.losses import DiceLoss
from kornia.metrics import mean_iou
from typing import Dict

from .base_model import BaseModel


class SegModel(BaseModel):
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.classes = cfg['classes']
        if self.mode == 'train':
            os.makedirs(os.path.join(self.checkpoint_dir, 'network'), exist_ok=True)
            # Set optimizers
            self._set_optimizer()
            os.makedirs(os.path.join(self.checkpoint_dir, 'optimizer'), exist_ok=True)
            # Set lr_scheduler
            self._set_lr_scheduler()
            os.makedirs(os.path.join(self.checkpoint_dir, 'lr_scheduler'), exist_ok=True)
            # Set Loss function
            self._set_loss_fn()
            self.train_loss = {}
            self.val_loss = {}
            self.train_metrics = {}
            self.val_metrics = {}
        elif self.mode == 'test':
            self.checkpoint_dir = cfg['checkpoint_dir']
            self.result_dir = cfg['result_dir']

    def _set_optimizer(self):
        params = self.network.parameters()
        optimizer = self.cfg['optimizer']
        if optimizer['name'] == 'adam':
            self.optimizer = torch.optim.Adam(
                params, lr=optimizer['lr'], weight_decay=optimizer['weight_decay'])
        elif optimizer['name'] == 'sgd':
            self.optimizer = torch.optim.SGD(
                params, lr=optimizer['lr'], weight_decay=optimizer['weight_decay'])
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
        self.dice_loss_fn = DiceLoss().to(self.device)
        self.lambda_ce = self.cfg['lambda_ce']
        self.lambda_dice = self.cfg['lambda_dice']

    def load_network_state(self, state_name: str):
        state_path = os.path.join(self.checkpoint_dir, 'network', state_name)
        self.network.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded network weights from {}.'.format(
                state_path
            ))

    def load_optimizer_state(self, state_name: str):
        state_path = os.path.join(self.checkpoint_dir, 'optimizer', state_name)
        self.optimizer.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded optimizer state from {}.'.format(
                state_path
            ))

    def load_lr_scheduler_state(self, state_name: str):
        state_path = os.path.join(self.checkpoint_dir, 'lr_scheduler', state_name)
        self.lr_scheduler.load_state_dict(torch.load(state_path))
        if self.logger:
            self.logger.info('Loaded lr_scheduler state from {}.'.format(
                state_path
            ))

    def train(self, train_dl: DataLoader, val_dl: DataLoader):
        assert self.mode == 'train', f"The mode must be 'train', but got {self.mode}"
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
            for i, batch in enumerate(train_dl):
                # train one batch
                self.train_one_batch(batch)
                # validation
                if (iteration_index % self.val_interval == 0) or (i == len(train_dl)-1):
                    val_batch = next(iter(val_dl))
                    self.validate_one_batch(val_batch, iteration_index)
                    self.write_tensorboard(iteration_index)
            
                    if self.logger:
                        self.logger.info(
                            "[iteration: {:d}, lr: {:f}] [Epoch {:d}/{:d}, batch {:d}/{:d}] [train_loss: {:.3f}, val_loss: {:.3f}]".format(
                            iteration_index, self.optimizer.param_groups[0]['lr'],
                            epoch, self.start_epoch + self.num_epochs-1, i, len(train_dl)-1,
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
            saved_path = os.path.join(self.checkpoint_dir, 'network', "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.checkpoint_dir, 'network', "{:d}.pth".format(epoch))
        torch.save(self.network.state_dict(), saved_path)
        if self.logger:
            self.logger.info("Saved network weights into {}".format(saved_path))

    def save_optimizer_state(self, epoch: int):
        load_prefix = self.cfg.get('load_prefix', None)
        save_prefix = self.cfg.get('save_prefix', None)
        if not save_prefix:
            save_prefix = load_prefix
        if save_prefix:
            saved_path = os.path.join(self.checkpoint_dir, 'optimizer', "{}_{:d}.pth".format(save_prefix, epoch))
        else:
            saved_path = os.path.join(self.checkpoint_dir, 'optimizer', "{:d}.pth".format(epoch))
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
        metrics['mIoU'] = mean_iou(
            F.softmax(pred_masks, dim=1).argmax(1),
            ref_masks,
            len(self.classes)).mean(1).mean(0)

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
    
    def test(self, test_dl: DataLoader, epoch: int, test_name: str, load_prefix=None):
        assert self.mode == 'test', f"The mode must be 'test', but got {self.mode}"
        
        weights_name = f"{load_prefix}_{epoch}" if load_prefix else f"{epoch}"
        self.load_network_state(f"{weights_name}.pth")
        result_dir = os.path.join(self.result_dir, test_name, weights_name)
        if os.path.exists(result_dir):
            shutil.rmtree(result_dir)
        os.makedirs(result_dir)
        os.makedirs(os.path.join(result_dir, 'paired'))

        t_elapse_list = []
        mIoU_list = []
        idx = 1
        for batch in tqdm(test_dl):
            inp_imgs = batch['img'].to(self.device)
            ref_masks = batch['mask'].to(self.device) if 'mask' in batch else None
            img_names = batch['img_name']
            num = len(inp_imgs)
            with torch.no_grad():
                self.network.eval()
                t_start = time.time()
                pred_masks = self.network(inp_imgs)
                
                # average inference time consumed by one batch
                t_elapse_avg = (time.time() - t_start) / num
                t_elapse_list.append(t_elapse_avg)

                # calculate mIoU
                mIoU = mean_iou(F.softmax(pred_masks, dim=1).argmax(1),
                    ref_masks, len(self.classes)).mean(1).mean(0)
                mIoU_list.append(mIoU.cpu().item())

                # record visual results and metrics values
                full_img = self._gen_comparison_img(inp_imgs, pred_masks, ref_masks)
                full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(result_dir, 'paired', f'{idx:06d}.png'), full_img)
                with open(os.path.join(result_dir, 'paired', f"{idx:06d}.txt"), 'w') as f:
                    f.write('\n'.join(img_names))
            idx += 1

        frame_rate = 1 / (sum(t_elapse_list) / len(t_elapse_list))
        mIoU = sum(mIoU_list) / len(mIoU_list)
        if self.logger:
            self.logger.info(
                '[epoch: {:d}] [framte_rate: {:.1f} fps, mIoU: {:.3f}]'.format(
                    epoch, frame_rate, mIoU
                )
            )
    
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