import sys
from networks.ranet import RANet
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.ndimage import zoom
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import yaml
from tqdm import tqdm

from data import (
    create_test_dataset, create_test_dataloader
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

m = RANet(3, 3, 15, 2).to(device)
m.load_state_dict(torch.load('./checkpoints/ie5/ra/LSUI_03/weights_299.pth'))
m.eval()
_, eval_nodes = get_graph_node_names(m)
# print(m)
# print(eval_nodes)
return_nodes = {
    'adm.0.3.sa.sigmoid': 'sa1',
    'adm.1.3.sa.sigmoid': 'sa2',
}
m = create_feature_extractor(m, return_nodes)

# Dataset config
ds_cfg = './configs/dataset/lsui.yaml'
with open(ds_cfg) as f:
    ds_cfg = yaml.load(f, yaml.FullLoader)

# Data pipeline
test_ds_type = ds_cfg['test'].get('type', None)
test_ds = create_test_dataset(test_ds_type, ds_cfg['test'])
test_dl_cfg = {
    'batch_size': 4,
    'shuffle': False,
    'num_workers': 4,
}
test_dl = create_test_dataloader(test_ds, test_dl_cfg)

res_dir = './results/ie5/ra/LSUI_03/LSUI/epoch_299/att_visualization'

idx = 1
for batch in tqdm(test_dl):
    imgs = batch['inp'].to(device)  # N,C,H,W
    with torch.no_grad():
        atts = m(imgs)
    sa1_mps = atts['sa1'].detach().cpu().numpy()
    sa1_mps = np.squeeze(sa1_mps, axis=1)  # N,H,W
    sa2_mps = atts['sa2'].detach().cpu().numpy()
    sa2_mps = np.squeeze(sa2_mps, axis=1)

    # visualize
    imgs_num = imgs.shape[0]
    n_row = 2
    fig, axes = plt.subplots(n_row, imgs_num, figsize=(2.56*imgs_num, 2.56*n_row))
    for ax in axes.flatten():
        ax.axis('off')
    for i in range(imgs_num):
        img = imgs[i].detach().cpu().numpy().transpose(1,2,0)
        sa1 = sa1_mps[i]
        sa2 = sa2_mps[i]
        scale1 = [im_sz/mp_sz for im_sz,mp_sz in zip(img.shape[0:-1], sa1.shape)]
        scale2 = [im_sz/mp_sz for im_sz,mp_sz in zip(img.shape[0:-1], sa2.shape)]
        axes[0][i].imshow(img)
        axes[0][i].imshow(zoom(sa1, zoom=scale1), cmap='jet', alpha=0.4)
        axes[1][i].imshow(img)
        axes[1][i].imshow(zoom(sa2, zoom=scale2), cmap='jet', alpha=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, f'{idx:06d}.png'))
    fig.clear()
    with open(os.path.join(res_dir, f"{idx:06d}.txt"), 'w') as f:
        f.write('\n'.join(batch['img_name']))
    idx += 1
    # break