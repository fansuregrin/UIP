import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from glob import glob
from PIL import Image, ImageDraw
from collections import OrderedDict


def gen_comparison(
        img_name_list,
        img_dirs,
        font_cfg=None,
        expected_size=(256, 256),
        title_y=-1.0,
        w_pad=None,
        h_pad=None,
        save_fig=False,
        save_folder=None,
        save_fmt='png',
        save_name='comparison'):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig_width = num_cols * (expected_size[0]/100*(1+0.1))
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
            ax = axes[i, j]
            ax.axis('off')
            if i == num_rows-1:
                ax.set_title(label, fontdict=font_cfg, y=title_y)
            ax.imshow(img)
    fig.tight_layout(w_pad=w_pad, h_pad=h_pad)
    if not save_fig:
        fig.show()
    if save_fig and os.path.exists(save_folder):
        fig.savefig(os.path.join(save_folder, f"{save_name}.{save_fmt}"), format=save_fmt)


def gen_comparison_with_local_mag(
        img_name_list,
        local_areas,
        img_dirs,
        font_cfg=None,
        expected_size=(256, 256),
        title_y=-0.2,
        outline_color=(255,255,0),
        outline_width=3,
        save_fig=False,
        save_folder=None,
        save_fmt='png',
        save_name='comparison'):
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
                ax.set_title(label, fontdict=font_cfg, y=title_y)
            ax.imshow(full_img)
    fig.tight_layout()
    if not save_fig:
        fig.show()
    if save_fig and os.path.exists(save_folder):
        fig.savefig(os.path.join(save_folder, f"{save_name}.{save_fmt}"), format=save_fmt)


def gen_comparison_with_local_mag2(
        img_name,
        local_area,
        img_dirs,
        font_size=28,
        expected_size=(256, 256),
        outline_color=(255,255,0),
        outline_width=3,
        save_fig=False,
        save_folder=None,
        save_fmt='png',
        save_name='comparison'):
    num = len(img_dirs)
    num_rows = math.floor(math.sqrt(num))
    num_cols = math.ceil(num/num_rows)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3*2, num_rows*3.5))

    label_font = {
        'fontfamily': 'Nimbus Roman',
        'fontsize': font_size,
    }

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
            ax.set_title(label, **label_font)
            ax.imshow(full_img)
    fig.tight_layout()
    if not save_fig:
        fig.show()
    if save_fig and os.path.exists(save_folder):
        fig.savefig(os.path.join(save_folder, f"{save_name}.{save_fmt}"), format=save_fmt)


def gen_comparison_with_local_edges(
        img_name_list,
        local_areas,
        img_dirs,
        font_cfg=None,
        expected_size=(256, 256),
        title_y=-0.2,
        outline_color=(255,255,0),
        outline_width=3,
        save_fig=False,
        save_folder=None,
        save_fmt='png',
        save_name='comparison'):
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
            ax = axes[i, j]
            ax.axis('off')
            if i == num_rows-1:
                ax.set_title(label, fontdict=font_cfg, y=title_y)
            ax.imshow(full_img)
    fig.tight_layout()
    if not save_fig:
        fig.show()
    if save_fig and os.path.exists(save_folder):
        fig.savefig(os.path.join(save_folder, f"{save_name}.{save_fmt}"), format=save_fmt)


def gen_comparison_with_local_edges2(
        img_name_list,
        local_areas,
        img_dirs,
        font_size=28,
        expected_size=(256, 256),
        outline_color=(255,255,0),
        outline_width=3,
        save_fig=False,
        save_folder=None,
        save_fmt='png',
        save_name='comparison'):
    num_rows = len(img_name_list)
    num_cols = len(img_dirs)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols*3*3, num_rows*3.5))

    label_font = {
        'fontfamily': 'Nimbus Roman',
        'fontsize': font_size,
    }

    if not isinstance(img_dirs, OrderedDict):
        img_dirs = OrderedDict(img_dirs)

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
            local_mag_rgb = np.asarray(local_mag, dtype=np.uint8)
            local_mag_gray = cv2.cvtColor(local_mag_rgb, cv2.COLOR_RGB2GRAY)

            local_mag = local_mag.resize(expected_size)
            draw_local_mag = ImageDraw.Draw(local_mag)
            draw_local_mag.rectangle(((0,0), local_mag.size), outline=outline_color, width=3)
            local_mag = np.asarray(local_mag, dtype=np.uint8)

            local_edges = cv2.Canny(local_mag_gray, 100, 200)
            local_edges = cv2.resize(local_edges, expected_size)
            local_edges = cv2.cvtColor(local_edges, cv2.COLOR_GRAY2RGB)
            local_edges = Image.fromarray(local_edges)
            draw_local_edges = ImageDraw.Draw(local_edges)
            draw_local_edges.rectangle(((0,0), local_edges.size), outline=outline_color, width=3)
            local_edges = np.asarray(local_edges, dtype=np.uint8)
            
            full_img = np.concatenate((img, local_mag, local_edges), 1)
            ax = axes[i, j]
            ax.axis('off')
            if i == 0:
                ax.set_title(label, **label_font)
            ax.imshow(full_img)
    fig.tight_layout()
    if not save_fig:
        fig.show()
    if save_fig and os.path.exists(save_folder):
        fig.savefig(os.path.join(save_folder, f"{save_name}.{save_fmt}"), format=save_fmt)


def get_random_img_names(img_dir, num):
    full_img_list = glob('*.jpg', root_dir=img_dir) + glob('*.png', root_dir=img_dir)
    return random.sample(full_img_list, num)


def gen_random_area(img_w, img_h, area_w, area_h):
    x1 = random.randint(0, img_w - area_w)
    y1 = random.randint(0, img_h - area_h)
    x2 = x1 + area_w
    y2 = y1 + area_h
    return (x1, y1, x2, y2)