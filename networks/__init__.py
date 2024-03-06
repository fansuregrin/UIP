from typing import Dict, Any

from .color_enhance import ColorEnhancementNet
from .ranet import (
    RANet, RANet2, RANet3, RANet4, RANet5
)
from .ege_unet import EGEUNet
from .fcn import FCN
from .unet import UNet
from .four_net import FourNet
from .mimo_unet import MIMOUNet
from .mimo_swinT_unet import (
    MIMOSwinTUNet, MIMOSwinTUNet2, MIMOSwinTUNet3, MIMOSwinTUNet4,
    MIMOSwinTUNet5
)
from .vit_enhancer import ViTEnhancer1, ViTEnhancer2


def create_network(cfg: Dict[str, Any]):
    name = cfg['name']
    if name == 'ce':
        net = ColorEnhancementNet()
    elif name == 'ra':
        net = RANet(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks'], cfg['n_down'],
            ngf = cfg['ngf'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up']
        )
    elif name == 'ra2':
        net = RANet2(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks'], cfg['n_down'],
            ngf = cfg['ngf'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            fmsm_kernel_size = cfg['fmsm_kernel_size'],
            fmsm_padding_size = cfg['fmsm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up']
        )
    elif name == 'ra3':
        net = RANet3(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks'], cfg['n_down'],
            ngf = cfg['ngf'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            fmsm_kernel_size = cfg['fmsm_kernel_size'],
            fmsm_padding_size = cfg['fmsm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up']
        )
    elif name == 'ra4':
        net = RANet4(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks_res'], cfg['n_blocks_wfef'], cfg['n_down'],
            ngf = cfg['ngf'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            fmsm_kernel_size = cfg['fmsm_kernel_size'],
            fmsm_padding_size = cfg['fmsm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up']
        )
    elif name == 'ra5':
        net = RANet5(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks_res'], cfg['n_down'],
            ngf = cfg['ngf'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            fmsm_kernel_size = cfg['fmsm_kernel_size'],
            fmsm_padding_size = cfg['fmsm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_wfef_down = cfg['use_wfef_down'],
            use_att_up = cfg['use_att_up'],
            use_wfef_up = cfg['use_wfef_up']
        )
    elif name == 'mimounet':
        net = MIMOUNet(
            num_res = cfg['num_res']
        )
    elif name == 'mimo_swinT_unet':
        net = MIMOSwinTUNet(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT']
        )
    elif name == 'mimo_swinT_unet2':
        net = MIMOSwinTUNet2(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT']
        )
    elif name == 'mimo_swinT_unet3':
        net = MIMOSwinTUNet3(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT']
        )
    elif name == 'mimo_swinT_unet4':
        net = MIMOSwinTUNet4(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT'],
            fused_window_process=cfg['fused_window_process']
        )
    elif name == 'mimo_swinT_unet5':
        net = MIMOSwinTUNet5(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT'],
            fused_window_process=cfg['fused_window_process']
        )
    elif name == 'vit_enhancer1':
        net = ViTEnhancer1(
            output_nc = cfg['output_nc'],
            n_up = cfg['n_up'],
            img_h = cfg['img_h'],
            img_w = cfg['img_w'],
            ngf = cfg['ngf'],
            patch_h = cfg['patch_h'],
            patch_w = cfg['patch_w'],
            vit_scale = cfg['vit_scale'],
            use_dropout = cfg['use_dropout'],
            use_att_up = cfg['use_att_up'],
            pretrained_encoder = cfg['pretrained_encoder']
        )
    elif name == 'vit_enhancer2':
        net = ViTEnhancer2(
            output_nc = cfg['output_nc'],
            n_up = cfg['n_up'],
            img_h = cfg['img_h'],
            img_w = cfg['img_w'],
            ngf = cfg['ngf'],
            patch_h = cfg['patch_h'],
            patch_w = cfg['patch_w'],
            vit_scale = cfg['vit_scale'],
            use_dropout = cfg['use_dropout'],
            use_att_up = cfg['use_att_up'],
            pretrained_encoder = cfg['pretrained_encoder']
        )
    elif name == 'fcn':
        net = FCN(
            num_classes = cfg['num_classes'],
            backbone = cfg['backbone']
        )
    elif name == 'ege_unet':
        net = EGEUNet(
            cfg['num_classes'],
            cfg['input_nc'],
            cfg['c_list'],
            cfg['bridge'],
            cfg['gt_ds']
        )
    elif name == 'unet':
        net = UNet(
            cfg['input_nc'],
            cfg['output_nc'],
            cfg['ngf'],
            cfg['norm_layer'],
            cfg['use_dropout']
        )
    elif name == 'four':
        net = FourNet(
            cfg['in_nc'],
            cfg['nc'],
            cfg['out_nc']
        )
    else:
        assert f"<{name}> is not supported!"

    return net