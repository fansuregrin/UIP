import argparse
import yaml
import torch
import os
import sys
from loguru import logger

from models import create_model
from data import (
    create_test_dataset, create_test_dataloader
)
from utils import (
    LOGURU_FORMAT
)


# Command-line options and arguments
parser = argparse.ArgumentParser()
parser.add_argument("--ds_cfg", type=str, default="configs/dataset/lsui.yaml")
parser.add_argument("--net_cfg", type=str, default="configs/network/ra_01.yaml")
parser.add_argument("--name", type=str, default="experiment", help="name of training process")
parser.add_argument('--test_name', type=str, help='name for test dataset')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='path to checkpoint dir')
parser.add_argument("--result_dir", type=str, default="results")
parser.add_argument("--batch_size", type=int, default=4, help="size of batches")
parser.add_argument("--epoch", type=int, nargs='+', default=99, help="which epoch to load")
args = parser.parse_args()

model_v = 'ie'

# Dataset config
with open(args.ds_cfg) as f:
    ds_cfg = yaml.load(f, yaml.FullLoader)

# Network config
with open(args.net_cfg) as f:
    net_cfg = yaml.load(f, yaml.FullLoader)

net_name = net_cfg['name']

# Create some useful directories
result_dir = "{}/{}/{}/{}".format(args.result_dir, model_v, net_name, args.name)
checkpoint_dir = "{}/{}/{}/{}/".format(args.checkpoint_dir, model_v, net_name, args.name)
log_dir = os.path.join(checkpoint_dir, 'logs/test')
os.makedirs(result_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize logger
logger.remove(0)
logger.add(sys.stdout, format=LOGURU_FORMAT)
logger.add(os.path.join(log_dir, "test_{time}.log"), format=LOGURU_FORMAT)

# Write some training infomation into log file
logger.info(f"Starting Training Process...")
logger.info(f"net_name: {net_name}")
logger.info(f"result_dir: {result_dir}")
logger.info(f"checkpoint_dir: {checkpoint_dir}")
logger.info(f"log_dir: {log_dir}")
for option, value in vars(args).items():
    logger.info(f"{option}: {value}")
for option, value in ds_cfg.items():
    logger.info(f"{option}: {value}")

# Set device for pytorch
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


# Data pipeline
test_ds = create_test_dataset('paired_img', ds_cfg['test'])
test_dl_cfg = {
    'batch_size': args.batch_size,
    'shuffle': False,
    'num_workers': 4,
}
test_dl = create_test_dataloader(test_ds, test_dl_cfg)

# Create and initialize model
model_cfg = {
    'mode': 'test',
    'device': DEVICE,
    'logger': logger,
    'result_dir': result_dir,
    'checkpoint_dir': checkpoint_dir,
    'name': args.name,
    'net_cfg': net_cfg,
}
model = create_model(model_v, model_cfg)

# Test pipeline
f = open(
    os.path.join(result_dir, args.test_name, 'metrics.csv'),
    'w')
f.write('epoch,frame_rate,psnr,ssim\n')
for epoch in args.epoch:
    model.load_weights(f'weights_{epoch}.pth')
    frame_rate, psnr, ssim = model.test(test_dl, epoch, args.test_name)
    f.write('{:d},{:.1f},{:.3f},{:.3f}\n'.format(
        epoch, frame_rate, psnr, ssim
    ))
f.close()