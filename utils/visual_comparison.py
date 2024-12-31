import os
import random
from glob import glob
from collections import OrderedDict
from typing import List, Dict, Any, Tuple, Optional
import math

import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib import gridspec
from matplotlib.axes import Axes
from PIL import Image, ImageDraw


SERIAL_NUMBERS = {
    'abc': tuple(chr(i) for i in range(97, 123)),
    'ABC': tuple(chr(i) for i in range(65, 91)),
    '123': tuple(i for i in range(1, 27))
}

DEFAULT_TITLE_CFG = {
    'fontfamily': 'serif',
    'verticalalignment': 'top',
    'y': -0.1
}


def _check_fp(root_dir: str, filename: str, fmts: Optional[List[str]] = None):
    fp = os.path.join(root_dir, filename)
    
    if not os.path.exists(fp) and fmts:
        for fmt in fmts:
            fp = os.path.join(root_dir, f'{filename}.{fmt}')
            if os.path.exists(fp): break
    
    if not os.path.exists(fp):
        raise FileNotFoundError(f"The specified file does not exist: '{fp}'")

    return fp


def gen_comparison(
    img_name_list: List[str],
    img_dirs: Dict[str, str],
    title_cfg: Dict[str, Any] = DEFAULT_TITLE_CFG,
    expected_size: Tuple[int, int] = (256, 256),
    auto_nb: str = 'abc',
    tight_layout_cfg: Dict = dict(),
    **kwargs
):
    """Multiple case comparisons of multiple images."""
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    for i in range(num_rows):
        for j, label in enumerate(img_dirs):
            img_fp = _check_fp(img_dirs[label], img_name_list[i],
                fmts=kwargs.get('fmts', None))
            img = Image.open(img_fp)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if img.size != expected_size:
                img = img.resize(expected_size,
                    resample=kwargs.get('resize_algo', Image.Resampling.BICUBIC))
            
            ax = fig.add_subplot(gs[i, j])
            ax.axis('off')
            if i == num_rows-1:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][j]}) {label}'
                ax.set_title(label, **title_cfg)
            ax.imshow(img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison2(
    img_name: str,
    img_dirs: Dict[str, str],
    title_cfg: Dict[str, Any] = DEFAULT_TITLE_CFG,
    n_row: int = None,
    expected_size: Tuple[int, int] = (256, 256),
    auto_nb: str = 'abc',
    tight_layout_cfg: Dict = dict(),
    **kwargs
):
    """Comparison of multiple cases in a single image."""
    num = len(img_dirs)
    if isinstance(n_row, int):
        num_rows = n_row
    else:
        num_rows = math.floor(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    for i,label in enumerate(img_dirs.keys()): 
            img_fp = _check_fp(img_dirs[label], img_name, fmts=kwargs.get('fmts', None))
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            img = np.asarray(img, dtype=np.uint8)
            ax = axes[i]
            if SERIAL_NUMBERS.get(auto_nb, None):
                label = f'({SERIAL_NUMBERS[auto_nb][i]}) {label}'
            ax.set_title(label, **title_cfg)
            ax.imshow(img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison3(
    img_name_list: List[str],
    img_dirs: Dict[str, str],
    title_cfg: Dict[str, Any] = None,
    expected_size: Tuple[int, int] = (256, 256),
    auto_nb: str = 'abc',
    tight_layout_cfg: Dict = dict(),
    **kwargs,
):
    if title_cfg is None:
        title_cfg = {
            'fontfamily': 'serif',
            'x': -0.1
        }

    num_rows = len(img_dirs)
    num_cols = len(img_name_list)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    for i,label in enumerate(img_dirs):
        for j in range(num_cols):
            img_fp = _check_fp(img_dirs[label], img_name_list[j],
                fmts=kwargs.get('fmts', None))
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            ax = axes[i, j]
            ax.axis('off')
            if j == 0:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][i]}) {label}'
                ax.set_title(label, va='center', ha='center', rotation='vertical',
                    **title_cfg)
            ax.imshow(img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison_with_local_mag(
    img_name_list,
    local_areas,
    img_dirs,
    title_cfg=DEFAULT_TITLE_CFG,
    expected_size=(256, 256),
    auto_nb='abc',
    outline_color=(255,255,0),
    outline_width=3,
    tight_layout_cfg=dict()
):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (2*expected_size[0]/100*(1+0.1)) 
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    if not isinstance(img_dirs, OrderedDict):
        img_dirs = OrderedDict(img_dirs)

    for i in range(num_rows):
        for j, label in enumerate(img_dirs):
            img_fp = os.path.join(img_dirs[label], img_name_list[i])
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            draw = ImageDraw.Draw(img)
            local_mag = img.crop(local_areas[i]).resize(expected_size)
            draw.rectangle(local_areas[i], outline=outline_color, width=outline_width)
            draw_local = ImageDraw.Draw(local_mag)
            draw_local.rectangle(((0,0), local_mag.size), outline=outline_color, width=3)
            img = np.asarray(img, dtype=np.uint8)
            local_mag = np.asarray(local_mag, dtype=np.uint8)
            full_img = np.concatenate((img, local_mag), 1)
            ax = axes[i, j]
            ax.axis('off')
            if i == num_rows-1:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][j]}) {label}'
                ax.set_title(label, **title_cfg)
            ax.imshow(full_img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison_with_local_mag2(
    img_name,
    local_area,
    img_dirs,
    title_cfg=DEFAULT_TITLE_CFG,
    expected_size=(256, 256),
    auto_nb='abc',
    outline_color=(255,255,0),
    outline_width=3,
    tight_layout_cfg=dict()
):
    num = len(img_dirs)
    num_rows = math.floor(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig_width = num_cols * (2*expected_size[0]/100*(1+0.1)) 
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    if not isinstance(img_dirs, OrderedDict):
        img_dirs = OrderedDict(img_dirs)

    axes = axes.flatten()
    for ax in axes:
        ax.axis('off')
    for i,label in enumerate(img_dirs.keys()): 
            img_fp = os.path.join(img_dirs[label], img_name)
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            draw = ImageDraw.Draw(img)
            local_mag = img.crop(local_area).resize(expected_size)
            draw.rectangle(local_area, outline=outline_color, width=outline_width)
            draw_local = ImageDraw.Draw(local_mag)
            draw_local.rectangle(((0,0), local_mag.size), outline=outline_color, width=3)
            img = np.asarray(img, dtype=np.uint8)
            local_mag = np.asarray(local_mag, dtype=np.uint8)
            full_img = np.concatenate((img, local_mag), 1)
            ax = axes[i]
            if SERIAL_NUMBERS.get(auto_nb, None):
                label = f'({SERIAL_NUMBERS[auto_nb][i]}) {label}'
            ax.set_title(label, **title_cfg)
            ax.imshow(full_img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison_with_local_mag3(
    img_name_list,
    local_areas,
    img_dirs,
    title_cfg=DEFAULT_TITLE_CFG,
    auto_nb='abc',
    expected_size=(256, 256),
    outline_color=(255,255,0),
    outline_width=3,
    tight_layout_cfg=dict()
):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    if num_rows == 1:
        axes = np.expand_dims(axes, 0)
    elif num_cols == 1:
        axes = np.expand_dims(axes, 1)

    if not isinstance(img_dirs, OrderedDict):
        img_dirs = OrderedDict(img_dirs)

    for i in range(num_rows):        
        for j, label in enumerate(img_dirs):
            if j == 0:
                # display the entire raw underwater image
                img_fp = os.path.join(img_dirs[label], img_name_list[i])
                img = Image.open(img_fp)
                if img.size != expected_size:
                    img = img.resize(expected_size)
                draw = ImageDraw.Draw(img)
                draw.rectangle(local_areas[i], outline=outline_color, width=outline_width)
            else:
                img_fp = os.path.join(img_dirs[label], img_name_list[i])
                img = Image.open(img_fp)
                if img.size != expected_size:
                    img = img.resize(expected_size)
                img = img.crop(local_areas[i]).resize(expected_size)
            img = np.asarray(img, dtype=np.uint8)
            ax = axes[i, j]
            ax.axis('off')
            if i == num_rows-1:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][j]}) {label}'
                ax.set_title(label, **title_cfg)
            ax.imshow(img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def gen_comparison_with_local_edges(
    img_name_list,
    local_areas,
    img_dirs,
    title_cfg=DEFAULT_TITLE_CFG,
    expected_size=(256, 256),
    auto_nb='abc',
    outline_color=(255,255,0),
    outline_width=3,
    tight_layout_cfg=dict()
):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (2*expected_size[0]/100*(1+0.1)) 
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs0 = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    for i in range(num_rows):
        for j, label in enumerate(img_dirs):
            img_fp = os.path.join(img_dirs[label], img_name_list[i])
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            
            draw = ImageDraw.Draw(img)
            local_mag = img.crop(local_areas[i])
            draw.rectangle(local_areas[i], outline=outline_color, width=outline_width)
            img = np.asarray(img, dtype=np.uint8)
            local_mag = np.asarray(local_mag, dtype=np.uint8)
            local_mag = cv2.cvtColor(local_mag, cv2.COLOR_RGB2GRAY)
            local_edges = cv2.Canny(local_mag, 100, 200)
            local_edges = cv2.resize(local_edges, expected_size)
            local_edges = cv2.cvtColor(local_edges, cv2.COLOR_GRAY2RGB)
            local_edges = Image.fromarray(local_edges)
            draw_local = ImageDraw.Draw(local_edges)
            draw_local.rectangle(((0,0), local_edges.size), outline=outline_color, width=3)
            local_edges = np.asarray(local_edges, dtype=np.uint8)
            full_img = np.concatenate((img, local_edges), 1)
    
            ax = fig.add_subplot(gs0[i, j])
            ax.axis('off')
            if i == num_rows-1:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][j]}) {label}'
                ax.set_title(label, **title_cfg)
            ax.imshow(full_img)
    
    fig.tight_layout(**tight_layout_cfg)
    
    return fig


def gen_comparison_with_local_edges2(
    img_name_list,
    local_areas,
    img_dirs,
    title_cfg=DEFAULT_TITLE_CFG,
    expected_size=(256, 256),
    auto_nb='abc',
    outline_color=(255,255,0),
    outline_width=3,
    tight_layout_cfg=dict()
):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (2*expected_size[0]/100*(1+0.1)) 
    fig_height = num_rows * (expected_size[1]/100*(1+0.1))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

    if not isinstance(img_dirs, OrderedDict):
        img_dirs = OrderedDict(img_dirs)

    for i in range(num_rows):
        for j, label in enumerate(img_dirs):
            img_fp = os.path.join(img_dirs[label], img_name_list[i])
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
  
            local_mag_pil = img.crop(local_areas[i])
            if j == 0:
                draw = ImageDraw.Draw(img)
                draw.rectangle(local_areas[i], outline=outline_color, width=outline_width)
            img = np.asarray(img, dtype=np.uint8)
            local_mag = np.asarray(local_mag_pil, dtype=np.uint8)
            local_mag_gray = cv2.cvtColor(local_mag, cv2.COLOR_RGB2GRAY)
            local_edges = cv2.Canny(local_mag_gray, 100, 200)
            local_edges = cv2.resize(local_edges, expected_size)
            local_edges = cv2.cvtColor(local_edges, cv2.COLOR_GRAY2RGB)
            local_edges = np.asarray(local_edges, dtype=np.uint8)

            if j == 0:
                full_img = np.concatenate((img, local_edges), 1)
            else:
                local_mag = np.asarray(local_mag_pil.resize(expected_size), np.uint8)
                full_img = np.concatenate((local_mag, local_edges), 1)
            ax = axes[i, j]
            ax.axis('off')
            if i == num_rows-1:
                if SERIAL_NUMBERS.get(auto_nb, None):
                    label = f'({SERIAL_NUMBERS[auto_nb][j]}) {label}'
                ax.set_title(label, **title_cfg)
            ax.imshow(full_img)
    fig.tight_layout(**tight_layout_cfg)
    return fig


def visualize_color_distribution(
        img_name,
        img_dirs,
        ref_dir,
        title_cfg=DEFAULT_TITLE_CFG,
        n_row=None,
        expected_size=(256, 256),
        w_idx = None,
        auto_nb='abc',
        tight_layout_cfg=dict()):
    num = len(img_dirs)
    if isinstance(n_row, int):
        num_rows = n_row
    else:
        num_rows = math.floor(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
    fig_height = num_rows * (expected_size[1]/100*(2+0.1))
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs0 = gridspec.GridSpec(num_rows, num_cols, figure=fig)

    ref = Image.open(os.path.join(ref_dir, img_name)).convert('RGB')
    if ref.size != expected_size:
        ref = ref.resize(expected_size)
    if w_idx is None: w_idx = ref.width // 2
    ref = np.asarray(ref, dtype=np.float32) / 255.0
    r_vals_ref = ref[:, w_idx, 0]

    for i,label in enumerate(img_dirs.keys()):
            img_fp = os.path.join(img_dirs[label], img_name)
            img = Image.open(img_fp)
            if img.size != expected_size:
                img = img.resize(expected_size)
            img_arr = np.asarray(img, dtype=np.float32) / 255.0

            sub_spec: gridspec.SubplotSpec = gs0[i // num_cols, i % num_cols]
            gs1 = sub_spec.subgridspec(2, 1)
            img_ax: Axes = fig.add_subplot(gs1[0])
            chart_ax: Axes = fig.add_subplot(gs1[1])
            
            draw = ImageDraw.Draw(img)
            draw.line([(w_idx, 0), (w_idx, img.height-1)], fill='blue', width=2)
            img_ax.imshow(img)
            img_ax.axis('off')
            
            chart_ax.set_ylim(0.0, 1.0)
            chart_ax.plot(r_vals_ref, label='Reference')
            chart_ax.plot(img_arr[:, w_idx, 0], label=label)
            chart_ax.axis('on')
            chart_ax.legend()
            if SERIAL_NUMBERS.get(auto_nb, None):
                label = f'({SERIAL_NUMBERS[auto_nb][i]}) {label}'
            chart_ax.set_title(label, **title_cfg)
    fig.tight_layout(**tight_layout_cfg)

    return fig


def get_random_img_names(img_dir, num):
    full_img_list = glob('*.jpg', root_dir=img_dir) + glob('*.png', root_dir=img_dir)
    return random.sample(full_img_list, num)


def gen_random_area(img_w, img_h, area_w, area_h):
    x1 = random.randint(0, img_w - area_w)
    y1 = random.randint(0, img_h - area_h)
    x2 = x1 + area_w
    y2 = y1 + area_h
    return (x1, y1, x2, y2)