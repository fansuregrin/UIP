import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.ansi_escape import *


parser = argparse.ArgumentParser(prog="Fetch full-reference metrics values from files")
parser.add_argument('results_dir', type=str, help='path to results directory')
parser.add_argument('--eval_type', type=str, default='ref', choices=['ref', 'nonref'])
parser.add_argument('--ds_names', type=str, nargs='+',
    default=['LSUI','EUVP515','UIEB100','OceanEx'])
parser.add_argument('--file_fmt', type=str, default='pkl', choices=['pkl', 'csv', 'xlsx'],
    help='extension name of the data file')
parser.add_argument('--metric_precision', type=int, default=3,
    help='float-point display precision of metric value')
parser.add_argument('--metric_names', type=str, nargs='+', default=['psnr', 'ssim'])
args = parser.parse_args()

read_methods = {
    'csv': pd.read_csv,
    'pkl': pd.read_pickle,
    'xlsx': pd.read_excel,
}

filename = f'{args.eval_type}_eval.{args.file_fmt}'
float_fmt = f'{{:.{args.metric_precision}f}}'.format
ds_names = args.ds_names
metric_names = args.metric_names

overall_df = pd.DataFrame(columns=['dataset',] + metric_names)
for ds_name in ds_names:
    target_fifle = f'{args.results_dir}/{ds_name}/{filename}'
    if not os.path.exists(target_fifle): continue
    read_fn = read_methods.get(args.file_fmt, None)
    if read_fn is None:
        print(f'{RED}cannot read .{args.file_fmt} file!{ENDSTYLE}')
        exit()
    df = read_fn(target_fifle)
    if df is None: continue
    
    row = {'dataset': ds_name}
    src_row = df[df['img_name'] == 'average']
    for metric in metric_names:
        row[metric] = src_row[metric].values[0]
    overall_df.loc[len(overall_df)] = row

overall_df.loc[len(overall_df)] = {'dataset': 'average'}
for metric in metric_names:
    overall_df.loc[len(overall_df)-1, metric] = overall_df[metric].mean()

print(f'reference eval of [{GREEN}{args.results_dir}/{{{','.join(ds_names)}}}{ENDSTYLE}]')
print(overall_df.to_string(index=False, float_format=float_fmt))