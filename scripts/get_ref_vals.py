import argparse
import os
import pandas as pd

GREEN="\033[32m"
RED="\033[31m"
BOLD="\033[1m"
BOLD_GREEN="\033[1;32m"
BOLD_BLUE="\033[1;34m"
ENDSTYLE="\033[0m"

refer_dict = {
    "EUVP515": "/DataA/pwz/workshop/Datasets/EUVP_Dataset/test_samples/GTr",
    "OceanEx": "/DataA/pwz/workshop/Datasets/ocean_ex/good",
    "UIEB100": "/DataA/pwz/workshop/Datasets/UIEB100/reference",
    "LSUI": "/DataA/pwz/workshop/Datasets/LSUI/test/ref",
}

metric_names = [
    'psnr', 'ssim', 'mse'
]

parser = argparse.ArgumentParser(prog="Fetch reference-based values from files")
# parser.add_argument('--model_v', type=str, required=True, help='model version')
# parser.add_argument('--net', type=str, required=True, help='network name')
# parser.add_argument('--name', type=str, required=True, help='checkpoint name')
# parser.add_argument('--epochs', nargs='+', type=int, required=True, help='epochs to fetch')
# parser.add_argument('--load_prefix', type=str, required=True, help='the prefix of weight file')
parser.add_argument('model_v', type=str, help='model version')
parser.add_argument('net', type=str, help='network name')
parser.add_argument('name', type=str, help='checkpoint name')
parser.add_argument('epochs', nargs='+', type=int, help='epochs to fetch')
parser.add_argument('load_prefix', type=str, help='the prefix of weight file')
parser.add_argument('--root_dir', type=str, default='./results', help='path to root directory of results')
args = parser.parse_args()

num_epoch = len(args.epochs)
if num_epoch > 1:
    info_str = 'reference eval of [{}{}/{}/{}/{}_{{{}}}{}]\n'.format(
        GREEN, args.model_v, args.net, args.name, args.load_prefix,
        ','.join(str(e) for e in args.epochs), ENDSTYLE
    )
    for ds_name in refer_dict:
        info_per_ds = f'{BOLD_BLUE}{ds_name}:{ENDSTYLE}\n'
        info_per_ds += ('=' * (len(metric_names)+1) * 10 + '\n')
        info_per_ds += '{}{:<10s}{:<10s}{:<10s}{:<10s}{}\n'.format(
            BOLD, 'epoch', 'psnr', 'ssim', 'mse', ENDSTYLE
        )
        info_per_ds += ('-' * (len(metric_names)+1) * 10 + '\n')
        metric_sums = {name:0.0 for name in metric_names}
        for epoch in args.epochs:
            target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/{args.name}/{ds_name}/{args.load_prefix}_{epoch}/ref_eval.csv'
            if not os.path.exists(target_fifle):
                continue
            df = pd.read_csv(target_fifle)
            avg_row = df[df['img_name'] == 'average']
            fmt_row_str = '{:<10}'.format(epoch)
            for metric in metric_names:
                val = avg_row[metric].values[0]
                metric_sums[metric] += val
                fmt_row_str += '{:<10.3f}'.format(val)
            info_per_ds += (fmt_row_str + '\n')
        avg_fmt_row_str = '{:<10}'.format('average')
        for metric in metric_names:
            avg_fmt_row_str += '{:<10.3f}'.format(metric_sums[metric]/num_epoch)
        info_per_ds += (avg_fmt_row_str + '\n')
        info_per_ds += ('=' * (len(metric_names)+1) * 10 + '\n\n')
        info_str += info_per_ds
    print(info_str)
else:
    epoch = args.epochs[0]
    info_str = 'reference eval of [{}{}/{}/{}/{}_{}{}]\n'.format(
        GREEN, args.model_v, args.net, args.name, args.load_prefix, epoch, ENDSTYLE
    )
    info_str += ('=' * (len(metric_names) * 10 + 15) + '\n')
    info_str += '{}{:<15s}{:<10s}{:<10s}{:<10s}{}\n'.format(
        BOLD, 'dataset', 'psnr', 'ssim', 'mse', ENDSTYLE
    )
    info_str += ('-' * (len(metric_names) * 10 + 15) + '\n')
    metric_sums = {name:0.0 for name in metric_names}
    for ds_name in refer_dict:
        target_fifle = f'{args.root_dir}/{args.model_v}/{args.net}/{args.name}/{ds_name}/{args.load_prefix}_{epoch}/ref_eval.csv'
        if not os.path.exists(target_fifle):
            continue
        df = pd.read_csv(target_fifle)
        avg_row = df[df['img_name'] == 'average']
        fmt_row_str = '{:<15}'.format(ds_name)
        for metric in metric_names:
            val = avg_row[metric].values[0]
            metric_sums[metric] += val
            fmt_row_str += '{:<10.3f}'.format(val)
        info_str += (fmt_row_str + '\n')
    avg_fmt_row_str = '{:<15}'.format('average')
    for metric in metric_names:
        avg_fmt_row_str += '{:<10.3f}'.format(metric_sums[metric]/len(refer_dict))
    info_str += (avg_fmt_row_str + '\n')
    info_str += ('=' * (len(metric_names) * 10 + 15) + '\n')
    print(info_str)