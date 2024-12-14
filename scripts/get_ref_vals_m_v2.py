import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from utils.ansi_escape import *


parser = argparse.ArgumentParser(prog="Fetch full-reference metrics values from files")
parser.add_argument('model_v', type=str, help='model version')
parser.add_argument('net', type=str, help='network name')
parser.add_argument('name', type=str, help='checkpoint name')
parser.add_argument('epochs', nargs='+', type=int, help='epochs to fetch')
parser.add_argument('--ds_names', type=str, nargs='+',
    default=['LSUI','EUVP515','UIEB100','OceanEx'])
parser.add_argument('--load_prefix', type=str, default='weights',
    help='the prefix of weight file')
parser.add_argument('--root_dir', type=str, default='./results',
    help='path to root directory of results')
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

float_fmt = f'{{:.{args.metric_precision}f}}'.format
num_epoch = len(args.epochs)
ds_names = args.ds_names
metric_names = args.metric_names
if num_epoch > 1:
    columns = ['epoch',] + metric_names
    ds_dfs = {ds: pd.DataFrame(columns=columns) for ds in ds_names}
    overall_df = pd.DataFrame(columns=['dataset',] + metric_names)
    for ds_name in ds_names:
        ds_df = ds_dfs[ds_name]
        for epoch in args.epochs:
            target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/\
{args.name}/{ds_name}/{args.load_prefix}_{epoch}/ref_eval.{args.file_fmt}'
            if not os.path.exists(target_fifle): continue
            read_fn = read_methods.get(args.file_fmt, None)
            if read_fn is None:
                print(f'{RED}cannot read .{args.file_fmt} file!{ENDSTYLE}')
                exit()
            df = read_fn(target_fifle)
            if df is None: continue
            
            row = {'epoch': epoch}
            src_row = df[df['img_name'] == 'average']
            for metric in metric_names:
                row[metric] = src_row[metric].values[0]
            ds_df.loc[len(ds_df)] = row
        ds_df.loc[len(ds_df)] = {'epoch': 'average'}
        overall_df.loc[len(overall_df)] = {'dataset': ds_name}
        for metric in metric_names:
            avg_val = ds_df[metric].mean()
            ds_df.loc[len(ds_df)-1, metric] = avg_val
            overall_df.loc[len(overall_df)-1, metric] = avg_val
    overall_df.loc[len(overall_df)] = {'dataset': 'average'}
    for metric in metric_names:
        overall_df.loc[len(overall_df)-1, metric] = overall_df[metric].mean()

    for ds in ds_dfs:
        print(f'{BOLD_BLUE}{ds}:{ENDSTYLE}')
        print(ds_dfs[ds].to_string(index=False, float_format=float_fmt))
        print()
    print(f"Average on [{BLUE}{','.join(ds_names)}{ENDSTYLE}]")
    print(overall_df.to_string(index=False, float_format=float_fmt))
else:
    overall_df = pd.DataFrame(columns=['dataset',] + metric_names)
    for ds_name in ds_names:
        target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/\
{args.name}/{ds_name}/{args.load_prefix}_{args.epochs[0]}/ref_eval.{args.file_fmt}'
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

    print(f'reference eval of [{GREEN}{args.root_dir}/{args.model_v}/\
{args.net}/{args.name}/{args.load_prefix}_{args.epochs[0]}{ENDSTYLE}]')
    print(overall_df.to_string(index=False, float_format=float_fmt))