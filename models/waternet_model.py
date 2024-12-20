import os
import time
import shutil
from typing import Any, Dict
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
from torchvision.utils import save_image

from losses import SemanticContentLoss as PerceptrualLoss
from models.img_enhance_model import ImgEnhanceModel
from models import _models


@_models.register('waternet')
class WaterNetModel(ImgEnhanceModel):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(cfg)

    def _set_loss_fn(self):
        self.mae_loss_fn = nn.L1Loss(reduction=self.cfg['l1_reduction']).to(self.device)
        self.percep_loss_fn = PerceptrualLoss().to(self.device)
        self.lambda_mae  = self.cfg['lambda_mae']
        self.lambda_percep = self.cfg['lambda_percep']

    def _calculate_loss(self, ref_imgs, pred_imgs, train=True):
        loss = self.train_loss if train else self.val_loss
        loss['mae'] = self.mae_loss_fn(pred_imgs, ref_imgs)
        loss['percep'] = self.percep_loss_fn(pred_imgs, ref_imgs)
        loss['total'] = self.lambda_mae * loss['mae'] + \
            self.lambda_percep * loss['percep']
        
    def train_one_batch(self, input_: Dict):
        inp_imgs = input_['inp'].to(self.device)
        inp_imgs_wb = input_['inp_wb'].to(self.device)
        inp_imgs_gc = input_['inp_gc'].to(self.device)
        inp_imgs_he = input_['inp_he'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        self.optimizer.zero_grad()
        self.network.train()
        pred_imgs = self.network(inp_imgs, inp_imgs_wb, inp_imgs_he, inp_imgs_gc)
        self._calculate_loss(ref_imgs, pred_imgs)
        self.train_loss['total'].backward()
        self.optimizer.step()
        self._calculate_metrics(ref_imgs, pred_imgs)

    def validate_one_batch(self, input_: Dict, iteration):
        inp_imgs = input_['inp'].to(self.device)
        inp_imgs_wb = input_['inp_wb'].to(self.device)
        inp_imgs_gc = input_['inp_gc'].to(self.device)
        inp_imgs_he = input_['inp_he'].to(self.device)
        ref_imgs = input_['ref'].to(self.device)
        with torch.no_grad():
            pred_imgs = self.network(inp_imgs, inp_imgs_wb, inp_imgs_he, inp_imgs_gc)
            self._calculate_loss(ref_imgs, pred_imgs, train=False)
            self._calculate_metrics(ref_imgs, pred_imgs, train=False)
            full_img = self._gen_comparison_img(inp_imgs, pred_imgs, ref_imgs)
            full_img = cv2.cvtColor(full_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.sample_dir, f'{iteration:06d}.png'), full_img)

    def test_one_epoch(self, epoch: int, test_name: str, load_prefix=None):
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
        for batch in tqdm(self.test_dl):
            inp_imgs = batch['inp'].to(self.device)
            inp_imgs_wb = batch['inp_wb'].to(self.device)
            inp_imgs_gc = batch['inp_gc'].to(self.device)
            inp_imgs_he = batch['inp_he'].to(self.device)
            ref_imgs = batch['ref'].to(self.device) if 'ref' in batch else None
            img_names = batch['img_name']
            num = len(inp_imgs)
            with torch.no_grad():
                self.network.eval()
                t_start = time.time()
                pred_imgs = self.network(inp_imgs, inp_imgs_wb, inp_imgs_he, inp_imgs_gc)
                
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

    @staticmethod
    def modify_args(parser, mode):
        if mode == 'train':
            parser.add_argument('--lambda_mae', type=float, default=1.0)
            parser.add_argument('--lambda_percep', type=float, default=0.5)
            parser.add_argument('--l1_reduction', type=str, default='mean')
        return parser