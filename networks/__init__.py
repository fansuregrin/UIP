from typing import Dict, Any

from .color_enhance import ColorEnhancementNet
from .ranet import RANet
from .ranet2 import RANet2
from .ranet3 import RANet3
from .ege_unet import EGEUNet
from .fcn import FCN
from .unet import UNet
from .four_net import FourNet


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