from segment_anything import (
    SamPredictor, SamAutomaticMaskGenerator,
    sam_model_registry
)
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import box_convert
import torchvision.transforms.functional as tvF
import torch
import argparse
import random
import glob
import os
import tqdm
import pandas as pd


def generate_unique_colors(num_colors):
    colors = set()
    while len(colors) < num_colors:
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        colors.add(color)
    return list(colors)

def segment(img, colors):
    with torch.no_grad():    
        masks = mask_generator.generate(img)
    
    result = {}
    
    img_tensor = torch.tensor(img.transpose(2, 0, 1))
    new_masks = torch.tensor(np.stack([m['segmentation'] for m in masks], 0))
    img_with_masks = draw_segmentation_masks(img_tensor, new_masks, colors=colors, alpha=0.8)
    result['seg_res'] = img_with_masks

    predicted_iou_list = [m['predicted_iou'] for m in masks]
    result['predicted_iou'] = sum(predicted_iou_list) / len(predicted_iou_list)

    stability_score_list = [m['stability_score'] for m in masks]
    result['stability_score'] = sum(stability_score_list) / len(stability_score_list)
    
    return result

parser = argparse.ArgumentParser('SAM infering')
parser.add_argument('--ckpt', type=str, default='./checkpoints/sam_pretrained/sam_vit_h_4b8939.pth')
parser.add_argument('--vit_scale', type=str, default='default')
parser.add_argument('--raw_dir', type=str, required=True)
parser.add_argument('--enh_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--expected_size', type=str, default='256x256')

if __name__ == '__main__':
    args = parser.parse_args()
    colors = generate_unique_colors(512)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry[args.vit_scale](checkpoint=args.ckpt).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    os.makedirs(args.save_dir, exist_ok=True)
    expected_size = [int(sz) for sz in args.expected_size.split('x')]
    assert len(expected_size) == 2, "invalid size"
    img_names = glob.glob('*.jpg', root_dir=args.raw_dir)
    img_names = img_names + glob.glob('*.png', root_dir=args.raw_dir)
    fig,axes = plt.subplots(nrows=1, ncols=4, squeeze=False, figsize=(4*expected_size[0]*0.01, 1*expected_size[1]*0.01*1.1))
    for ax in axes.flatten():
        ax.axis('off')

    columns = ['img_name', 'raw_iou', 'enh_iou', 'raw_stability', 'enh_stability']
    df = pd.DataFrame(columns=columns)

    for img_name in tqdm.tqdm(img_names):
        raw_img_fp = os.path.join(args.raw_dir, img_name)
        enh_img_fp = os.path.join(args.enh_dir, img_name)
        raw_pil_img = Image.open(raw_img_fp).resize(expected_size)
        raw_img = np.asarray(raw_pil_img)
        enh_pil_img = Image.open(enh_img_fp).resize(expected_size)
        enh_img = np.asarray(enh_pil_img)
        raw_result = segment(raw_img, colors)
        enh_result = segment(enh_img, colors)
        raw_seg_res = raw_result['seg_res']
        enh_seg_res = enh_result['seg_res']
        
        axes[0,0].imshow(raw_img)
        axes[0,0].set_title('raw')
        axes[0,1].imshow(raw_seg_res.cpu().numpy().transpose(1,2,0))
        axes[0,1].set_title('raw-seg')
        axes[0,2].imshow(enh_img)
        axes[0,2].set_title('enhanced')
        axes[0,3].imshow(enh_seg_res.cpu().numpy().transpose(1,2,0))
        axes[0,3].set_title('enhanced-seg')
        fig.tight_layout()
        fig.savefig(os.path.join(args.save_dir, img_name))

        row = {
            'img_name': img_name,
            'raw_iou': raw_result['predicted_iou'],
            'enh_iou': enh_result['predicted_iou'],
            'raw_stability': raw_result['stability_score'],
            'enh_stability': enh_result['stability_score']
        }
        df.loc[len(df)] = row
    
    row_avg = {
        'img_name': 'average',
        'raw_iou': df['raw_iou'].mean(),
        'enh_iou': df['enh_iou'].mean(),
        'raw_stability': df['raw_stability'].mean(),
        'enh_stability': df['enh_stability'].mean()
    }
    df.loc[len(df)] = row_avg

    csv_fp = os.path.join(args.save_dir, 'sam_record.csv')
    df.to_csv(csv_fp, index=False)