# author: Fansure Grin
# email: pwz113436@gmail.com
"""Full-Reference Evaluation.
metrics:
    - PSNR
    - SSIM
    - MSE
"""
import argparse
import os
from collections import OrderedDict
from glob import glob
from typing import overload

import torch
from torch.nn.functional import mse_loss
from torchvision.transforms.functional import to_tensor
import pandas as pd
from PIL import Image
from tqdm import tqdm
from kornia.metrics import psnr, ssim

from utils.ansi_escape import *
from utils.df_save import save_df


class RefMetric:
    def __init__(self, **kwargs):
        self.args = kwargs

    @overload
    def __call__(self, inp, ref):
        pass


class PSNR(RefMetric):
    def __call__(self, inp, ref):
        return psnr(inp, ref, **self.args).item()
    

class SSIM(RefMetric):
    def __call__(self, inp, ref):
        return ssim(inp, ref, **self.args).mean().item()
    

class MSE(RefMetric):
    def __call__(self, inp, ref):
        return mse_loss(inp, ref, **self.args).item()


if __name__ == '__main__':
    ## parse command line arguments and options
    parser = argparse.ArgumentParser()
    parser.add_argument('-inp', '--input_dir', required=True, type=str,
        help='path to folder of input images')
    parser.add_argument('-ref', '--refer_dir', required=True, type=str,
        help='path to folder of reference images')
    parser.add_argument('-out', '--output_dir', type=str, default='',
        help='path to folder of results')
    parser.add_argument('--resize', action='store_true',
        help='whether resize the input and reference images')
    parser.add_argument('--width', default=256, type=int,
        help='image width for resizing')
    parser.add_argument('--height', default=256, type=int,
        help='image height for resizing')
    parser.add_argument('-outfmt', '--output_format', type=str, 
        default=['csv', 'pkl'], nargs='+', choices=['csv', 'pkl', 'tex', 'xlsx'],
        help='the ouput format of evaluation results')
    parser.add_argument('--window_size', type=int, default=11,
        help='the window size for calculating ssim')
    args = parser.parse_args()
    if args.output_dir == '':
        args.output_dir = args.input_dir

    metrics = OrderedDict(
        psnr = PSNR(max_val=1.0),
        ssim = SSIM(window_size=args.window_size, max_val=1.0),
        mse  = MSE(reduction='mean'),
    )
    columns = ['img_name',] + list(metrics.keys())
    df = pd.DataFrame(columns=columns)

    ## prepare lists of image paths
    pred_imgs_dir = args.input_dir
    expected_size = (args.width, args.height)
    img_name_list = glob('*.png', root_dir=pred_imgs_dir)
    img_name_list.extend(glob('*.jpg', root_dir=pred_imgs_dir))
    img_name_list.sort()

    print(f'evaluating [{GREEN}{pred_imgs_dir}{ENDSTYLE}]...')
    for img_name in tqdm(img_name_list):
        pred_img = Image.open(os.path.join(pred_imgs_dir, img_name))
        ref_img = Image.open(os.path.join(args.refer_dir, img_name))
        if args.resize:
            if pred_img.size != expected_size:
                pred_img = pred_img.resize(expected_size)
            if ref_img.size != expected_size:
                ref_img = ref_img.resize(expected_size)
        pred_img = to_tensor(pred_img).unsqueeze(0)
        ref_img = to_tensor(ref_img).unsqueeze(0)
        row = {'img_name': img_name}
        assert len(pred_img.shape) == 4
        for metric_name, metric_fn in metrics.items():
            row[metric_name] = metric_fn(pred_img, ref_img)
        df.loc[len(df)] = row
    row_avg = {'img_name': 'average'}
    for name in metrics:
        row_avg[name] = df[name].mean()
    df.loc[len(df)] = row_avg

    ## save eval data
    save_name = 'ref_eval'
    for out_fmt in args.output_format:
        save_fp = os.path.join(args.output_dir, f'{save_name}.{out_fmt}')
        save_df(df, save_fp)
        print(f"Saved eval data into [{GREEN}{save_fp}{ENDSTYLE}]!")