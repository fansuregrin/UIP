import argparse
import os
import pandas as pd

GREEN="\033[32m"
RED="\033[31m"
BOLD="\033[1m"
BOLD_GREEN="\033[1;32m"
BOLD_BLUE="\033[1;34m"
ENDSTYLE="\033[0m"

ds_name_list = [
    "U45",
    "RUIE_Color90",
    "UPoor200",
    "UW2023",
]

metric_names = [
    'niqe', 'musiq', 'uranker', 'uciqe', 'uiqm'
]

read_methods = {
    'csv': pd.read_csv,
    'pkl': pd.read_pickle,
    'xlsx': pd.read_excel,
}

parser = argparse.ArgumentParser(prog="Fetch reference-based values from files")
parser.add_argument('model_v', type=str, help='model version')
parser.add_argument('net', type=str, help='network name')
parser.add_argument('name', type=str, help='checkpoint name')
parser.add_argument('epochs', nargs='+', type=int, help='epochs to fetch')
parser.add_argument('load_prefix', type=str, help='the prefix of weight file')
parser.add_argument('--root_dir', type=str, default='./results', help='path to root directory of results')
parser.add_argument('--file_ext', type=str, default='pkl', choices=['pkl', 'csv', 'xlsx'], help='extension name of the data file')
parser.add_argument('--metric_precision', type=int, default=3, help='float-point display precision of metric value')
args = parser.parse_args()

metric_val_fmt = '{{:<10.{}f}}'.format(args.metric_precision)

num_epoch = len(args.epochs)
if num_epoch > 1:
    info_str = 'non-reference eval of [{}{}/{}/{}/{}_{{{}}}{}]\n'.format(
        GREEN, args.model_v, args.net, args.name, args.load_prefix,
        ','.join(str(e) for e in args.epochs), ENDSTYLE
    )

    info_ds_avg = f'Average on [{BOLD_BLUE}{",".join(ds_name_list)}]:{ENDSTYLE}\n'
    info_ds_avg += ('=' * (len(metric_names) * 10 + 15) + '\n')
    info_ds_avg += '{}{:<15s}'.format(BOLD, 'dataset')
    for metric in metric_names:
        info_ds_avg += '{:<10s}'.format(metric)
    info_ds_avg += '{}\n'.format(ENDSTYLE)
    info_ds_avg += ('-' * (len(metric_names) * 10 + 15) + '\n')

    ds_metrics_sum = {name:0.0 for name in metric_names}
    for ds_name in ds_name_list:
        info_per_ds = f'{BOLD_BLUE}{ds_name}:{ENDSTYLE}\n'
        info_per_ds += ('=' * (len(metric_names)+1) * 10 + '\n')
        info_per_ds += '{}{:<10s}'.format(BOLD, 'epoch')
        for metric in metric_names:
            info_per_ds += '{:<10s}'.format(metric)
        info_per_ds += '{}\n'.format(ENDSTYLE)
        info_per_ds += ('-' * (len(metric_names)+1) * 10 + '\n')
        epoch_metric_sums = {name:0.0 for name in metric_names}
        for epoch in args.epochs:
            target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/{args.name}/{ds_name}/{args.load_prefix}_{epoch}/noref_eval.{args.file_ext}'
            if not os.path.exists(target_fifle):
                continue
            df = read_methods.get(args.file_ext, lambda: None)(target_fifle)
            avg_row = df[df['img_name'] == 'average']
            fmt_row_str = '{:<10}'.format(epoch)
            for metric in metric_names:
                val = avg_row[metric].values[0]
                epoch_metric_sums[metric] += val
                fmt_row_str += metric_val_fmt.format(val)
            info_per_ds += (fmt_row_str + '\n')
        avg_fmt_row_str = '{:<10}'.format('average')
        info_ds_avg += '{:<15s}'.format(ds_name)
        for metric in metric_names:
            epoch_avg_val = epoch_metric_sums[metric]/num_epoch
            ds_metrics_sum[metric] += epoch_avg_val
            avg_fmt_row_str += metric_val_fmt.format(epoch_avg_val)
            info_ds_avg += metric_val_fmt.format(epoch_avg_val)
        info_ds_avg += '\n'
        info_per_ds += (avg_fmt_row_str + '\n')
        info_per_ds += ('=' * (len(metric_names)+1) * 10 + '\n\n')
        info_str += info_per_ds
    
    info_ds_avg += '{:<15s}'.format('average')
    for metric in metric_names:
        info_ds_avg += metric_val_fmt.format(ds_metrics_sum[metric]/len(ds_name_list))
    info_ds_avg += '\n'
    info_ds_avg += ('=' * (len(metric_names) * 10 + 15) + '\n')

    info_str += info_ds_avg
    print(info_str)
else:
    epoch = args.epochs[0]
    info_str = 'reference eval of [{}{}/{}/{}/{}_{}{}]\n'.format(
        GREEN, args.model_v, args.net, args.name, args.load_prefix, epoch, ENDSTYLE
    )
    info_str += ('=' * (len(metric_names) * 10 + 15) + '\n')
    info_str += '{}{:<15s}'.format(BOLD, 'dataset')
    for metric in metric_names:
        info_str += '{:<10s}'.format(metric)
    info_str += '{}\n'.format(ENDSTYLE)
    info_str += ('-' * (len(metric_names) * 10 + 15) + '\n')
    metric_sums = {name:0.0 for name in metric_names}
    for ds_name in ds_name_list:
        target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/{args.name}/{ds_name}/{args.load_prefix}_{epoch}/noref_eval.{args.file_ext}'
        if not os.path.exists(target_fifle):
            continue
        df = read_methods.get(args.file_ext, lambda: None)(target_fifle)
        avg_row = df[df['img_name'] == 'average']
        fmt_row_str = '{:<15}'.format(ds_name)
        for metric in metric_names:
            val = avg_row[metric].values[0]
            metric_sums[metric] += val
            fmt_row_str += metric_val_fmt.format(val)
        info_str += (fmt_row_str + '\n')
    avg_fmt_row_str = '{:<15}'.format('average')
    for metric in metric_names:
        avg_fmt_row_str += metric_val_fmt.format(metric_sums[metric]/len(ds_name_list))
    info_str += (avg_fmt_row_str + '\n')
    info_str += ('=' * (len(metric_names) * 10 + 15) + '\n')
    print(info_str)