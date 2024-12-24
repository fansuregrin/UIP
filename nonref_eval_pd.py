import os
from glob import glob
import argparse
from collections import OrderedDict
from typing import overload

import torch
import pyiqa
import pandas as pd
from tqdm import tqdm

from utils.uciqe import getUCIQE
from utils.uiqm import getUIQM
from utils.ansi_escape import *
from utils.df_save import save_df


class NonRefMetric:
    def __init__(self, **kwargs):
        self.args = kwargs
    
    @overload
    def __call__(self, inp):
        pass


class NIQE(NonRefMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = pyiqa.create_metric('niqe', **self.args)

    def __call__(self, inp):
        return self.metric(inp).item()


class MUSIQ(NonRefMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = pyiqa.create_metric('musiq', **self.args)

    def __call__(self, inp):
        return self.metric(inp).item()


class URanker(NonRefMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric = pyiqa.create_metric('uranker', **self.args)

    def __call__(self, inp):
        return self.metric(inp).item()


class UCIQE(NonRefMetric):
    def __call__(self, inp):
        return getUCIQE(inp)


class UIQM(NonRefMetric):
    def __call__(self, inp):
        return getUIQM(inp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-inp', '--input_dir', type=str, required=True)
    parser.add_argument('-out', '--output_dir', type=str, default='')
    parser.add_argument('-outfmt', '--output_format', type=str,
        default=['csv', 'pkl'], nargs='+', choices=['csv', 'pkl', 'tex', 'xlsx'],
        help='the ouput format of evaluation results')
    args = parser.parse_args()
    if args.output_dir == '':
        args.output_dir = args.input_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## define metrics and dataframe
    metrics = OrderedDict(
        niqe  = NIQE(device=device),
        musiq = MUSIQ(device=device),
        uranker = URanker(device=device),
        uciqe = UCIQE(),
        uiqm = UIQM(),
    )
    columns = ['img_name',] + list(metrics.keys())
    df = pd.DataFrame(columns=columns)

    ## prepare lists of image paths
    input_dir = args.input_dir
    if not os.path.exists(input_dir):
        print(f"{RED}[{input_dir}] not exist!{ENDSTYLE}")
        exit()
    img_name_list = glob('*.png', root_dir=input_dir)
    img_name_list.extend(glob('*.jpg', root_dir=input_dir))
    img_name_list.sort()

    print(f'evaluating [{GREEN}{input_dir}{ENDSTYLE}]...')
    for img_name in tqdm(img_name_list):
        img_path = os.path.join(input_dir, img_name)
        row = {'img_name': img_name}
        for metric_name, metric_fn in metrics.items():
            row[metric_name] = metric_fn(img_path)
        df.loc[len(df)] = row
    row_avg = {'img_name': 'average'}
    for name in metrics:
        row_avg[name] = df[name].mean()
    df.loc[len(df)] = row_avg

    ## save eval data
    save_name = 'nonref_eval'
    for out_fmt in args.output_format:
        save_fp = os.path.join(args.output_dir, f'{save_name}.{out_fmt}')
        save_df(df, save_fp)
        print(f"Saved eval data into [{GREEN}{save_fp}{ENDSTYLE}]!")