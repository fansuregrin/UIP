import argparse
import os
from glob import glob
from collections import OrderedDict

import torch
import pyiqa
from PIL import Image
from torchvision.transforms.functional import to_tensor
from kornia.color import rgb_to_ycbcr
from tqdm import tqdm

from utils.uciqe import getUCIQE
from utils.uiqm import getUIQM
from utils.ansi_escape import *


parser = argparse.ArgumentParser()
parser.add_argument('-inp', '--input_dir', type=str, required=True)
parser.add_argument('-out', '--output_dir', type=str, default='')
args = parser.parse_args()
if args.output_dir == '':
    args.output_dir = args.input_dir

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

niqe = pyiqa.create_metric('niqe', device=device)
musiq = pyiqa.create_metric('musiq', device=device)
uranker = pyiqa.create_metric('uranker', device=device)
metrics = OrderedDict(
    niqe  = {'fn': niqe, 'val': 0.0},
    musiq = {'fn': musiq, 'val': 0.0},
    uranker = {'fn': uranker, 'val': 0.0},
    uciqe = {'fn': getUCIQE, 'val':0.0},
    uiqm = {'fn': getUIQM, 'val': 0.0},
)

input_imgs_dir = args.input_dir
img_name_list = glob('*.png', root_dir=input_imgs_dir)
img_name_list.extend(glob('*.jpg', root_dir=input_imgs_dir))
img_name_list.sort()

print(f'evaluating [{GREEN}{args.input_dir}{GREEN}]...')
noref_f = open(os.path.join(args.input_dir, 'nonref_eval.csv'), 'w')
noref_f.write('img_name,{}\n'.format(','.join(metrics.keys())))
for img_name in tqdm(img_name_list):
    img_path = os.path.join(input_imgs_dir, img_name)
    img = to_tensor(Image.open(img_path)).unsqueeze(0)
    vals = []
    assert len(img.shape) == 4
    for metric_name, metric in metrics.items():
        if metric_name == 'niqe':
            val = metric['fn'](rgb_to_ycbcr(img)).item()
        elif metric_name == 'uciqe' or metric_name == 'uiqm':
            val = metric['fn'](img_path)
        else:
            val = metric['fn'](img).item()
        metric['val'] += val
        vals.append('{:.3f}'.format(val))
    noref_f.write('{},{}\n'.format(img_name, ','.join(vals)))
avg_vals = ['{:.3f}'.format(metrics[name]['val']/len(img_name_list)) for name in metrics]
noref_f.write('average,{}\n'.format(','.join(avg_vals)))
noref_f.close()

print('{}'.format('\t'.join(metrics.keys())))
print('{}'.format('\t'.join(avg_vals)))