import torch
import pyiqa
import argparse
import os
import pandas as pd
from PIL import Image
from collections import OrderedDict
from torchvision.transforms.functional import to_tensor
from kornia.color import rgb_to_ycbcr
from tqdm import tqdm

from utils.uciqe_uiqm import getUCIQE, getUIQM


GREEN="\033[32m"
RED="\033[31m"
BOLD="\033[1m"
BOLD_GREEN="\033[1;32m"
BOLD_BLUE="\033[1;34m"
ENDSTYLE="\033[0m"

parser = argparse.ArgumentParser()
parser.add_argument('-re_dir', '--results_dir', type=str, default='results')
parser.add_argument('--model_v', type=str, default='ie')
parser.add_argument('--net', type=str, default='ra')
parser.add_argument('--name', type=str)
parser.add_argument('--ds_name', type=str)
parser.add_argument('--epochs', type=int, nargs='+')
parser.add_argument("--load_prefix", type=str, default='weights', help="the prefix string of the filename of the weights to be loaded")

args = parser.parse_args()

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

columns = ['img_name',] + list(metrics.keys())

if not isinstance(args.epochs, list):
    args.epochs = [args.epochs]

for epoch in args.epochs:
    results_dir = os.path.join(args.results_dir, args.model_v, args.net,
                                 args.name, args.ds_name, f'{args.load_prefix}_{epoch}')
    if not os.path.exists(results_dir):
        print(f"{RED}[{results_dir}] not exist!{ENDSTYLE}")
        continue
    print(f'evaluating [{results_dir}]...')
    pred_imgs_dir = os.path.join(results_dir, 'single/predicted')
    img_name_list = os.listdir(pred_imgs_dir)
    img_name_list.sort()
    df = pd.DataFrame(columns=columns)
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(pred_imgs_dir, img_name)
        img = to_tensor(Image.open(img_path)).unsqueeze(0)
        row = {'img_name': img_name}
        assert len(img.shape) == 4
        for metric_name, metric in metrics.items():
            if metric_name == 'niqe':
                val = metric['fn'](rgb_to_ycbcr(img)).item()
            elif metric_name == 'uciqe' or metric_name == 'uiqm':
                val = metric['fn'](img_path)
            else:
                val = metric['fn'](img).item()
            metric['val'] += val
            row[metric_name] = val
        df.loc[len(df)] = row
    row_avg = {'img_name': 'average'}
    for name in metrics:
        row_avg[name] = metrics[name]['val']/len(img_name_list)
    df.loc[len(df)] = row_avg
    for metric_name in metrics:
        metrics[metric_name]['val'] = 0.0
    csv_fp = os.path.join(results_dir, 'noref_eval.csv')
    pkl_fp = os.path.join(results_dir, 'noref_eval.pkl')
    df.to_csv(csv_fp, index=False)
    df.to_pickle(pkl_fp)
    print(f"Saved eval data into [{GREEN}{csv_fp}{ENDSTYLE}] and [{GREEN}{pkl_fp}{ENDSTYLE}]!")