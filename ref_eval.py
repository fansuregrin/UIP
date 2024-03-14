import argparse
import os
import torch
from PIL import Image
from collections import OrderedDict
from torchvision.transforms.functional import to_tensor
from glob import glob
from tqdm import tqdm
from kornia.metrics import psnr, ssim
from torch.nn.functional import mse_loss
from functools import partial


parser = argparse.ArgumentParser()
parser.add_argument('-inp', '--input_dir', type=str, help='path to folder of input images')
parser.add_argument('-ref', '--refer_dir', type=str, help='path to folder of reference images')
parser.add_argument('-out', '--output_dir', type=str, help='path to folder of results')
parser.add_argument('--resize', action='store_true', help='whether resize the input and reference images')
parser.add_argument('--width', default=256, type=int, help='image width for resizing')
parser.add_argument('--height', default=256, type=int, help='image height for resizing')
args = parser.parse_args()

args = parser.parse_args()

metrics = OrderedDict(
    psnr  = {'fn': partial(psnr, max_val=1.0), 'val': 0.0},
    ssim = {'fn': partial(ssim, window_size=11, max_val=1.0), 'val': 0.0},
    mse = {'fn': partial(mse_loss, reduction='mean'), 'val': 0.0},
)

pred_imgs_dir = args.input_dir
expected_sized = (args.width, args.height)
print(f'evaluating [{pred_imgs_dir}]...')
img_name_list = glob('*.png', root_dir=pred_imgs_dir)
img_name_list.extend(glob('*.jpg', root_dir=pred_imgs_dir))
ref_f = open(os.path.join(args.output_dir, 'ref_eval.csv'), 'w')
ref_f.write('img_name,{}\n'.format(','.join(metrics.keys())))
img_name_list.sort()
for img_name in tqdm(img_name_list):
    pred_img = Image.open(os.path.join(pred_imgs_dir, img_name))
    ref_img = Image.open(os.path.join(args.refer_dir, img_name))
    if args.resize:
        if pred_img.size != expected_sized:
            pred_img = pred_img.resize(expected_sized)
        if ref_img.size != expected_sized:
            ref_img = ref_img.resize(expected_sized)
    pred_img = to_tensor(pred_img).unsqueeze(0)
    ref_img = to_tensor(ref_img).unsqueeze(0)
    vals = []
    assert len(pred_img.shape) == 4
    for metric_name, metric in metrics.items():
        val = metric['fn'](pred_img, ref_img)
        if metric_name == 'ssim':
            val = torch.mean(val).item()
        else:
            val = val.item()
        metric['val'] += val
        vals.append('{:.3f}'.format(val))
    ref_f.write('{},{}\n'.format(img_name, ','.join(vals)))
avg_vals = ['{:.3f}'.format(metrics[name]['val']/len(img_name_list)) for name in metrics]
ref_f.write('average,{}\n'.format(','.join(avg_vals)))
ref_f.close()

print('{}'.format('\t'.join(metrics.keys())))
print('{}'.format('\t'.join(avg_vals)))