import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from torchvision.utils import draw_segmentation_masks
from torchmetrics.classification import MulticlassJaccardIndex

from data import (
    create_train_dataset, create_train_dataloader
)


def calc_mIoU(pred, label, num_classes):
    pred = F.softmax(pred, dim=1)
    pred = pred.argmax(1).squeeze(1)
    label = label.argmax(1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = label.view(-1)
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() +\
                    target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)

ds_cfg_fp = 'configs/dataset/suim.yaml'
ds_cfg = yaml.load(open(ds_cfg_fp, 'r'), yaml.FullLoader)
train_ds = create_train_dataset('seg', ds_cfg['train'])
train_dl_cfg = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4
}
train_dl = create_train_dataloader(train_ds, train_dl_cfg)

val_ds = create_train_dataset('seg', ds_cfg['val'])
val_dl_cfg = {
    'batch_size': 4,
    'shuffle': True,
    'num_workers': 4
}
val_dl = create_train_dataloader(val_ds, val_dl_cfg)

color_map = ds_cfg['train']['color_map']
colors = ['#'+c for c in sorted(color_map.keys())]


# for batch in train_dl:
#     imgs = batch['img']
#     masks = batch['mask']
#     imgs_with_mask = []
#     for img, mask in zip(imgs, masks):
#         img = (img * 255).to(torch.uint8)
#         normalized_mask = F.softmax(mask, dim=0)
#         boolean_mask = torch.stack([(mask.argmax(0) == i) for i in range(len(color_map))])
#         img_with_mask = draw_segmentation_masks(img, boolean_mask, alpha=0.5, colors=colors)
#         imgs_with_mask.append(img_with_mask)
#     break


for batch in train_dl:
    imgs = batch['img']
    masks = batch['mask']
    print(calc_mIoU(masks, masks, 8))
    break