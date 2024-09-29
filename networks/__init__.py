from typing import Dict, Any

from .color_enhance import ColorEnhancementNet
from .ranet import (
    RANet, RANet2, RANet3, RANet4, RANet5, RANet6
)
from .ege_unet import EGEUNet
from .fcn import FCN
from .unet import UNet
from .deeplab import DeepLabV3
from .four_net import FourNet
from .mimo_unet import MIMOUNet
from .mimo_swinT_unet import (
    MIMOSwinTUNet, MIMOSwinTUNet2, MIMOSwinTUNet3, MIMOSwinTUNet4,
    MIMOSwinTUNet5, MIMOSwinTUNet6, MIMOSwinTUNet7
)
from .vit_enhancer import ViTEnhancer1, ViTEnhancer2
from .vm_unet.vmunet import VMUNet
from .vg_unet import VGUNet, VGUNet2, VGUNet3, VGUNet4
from .erd import ERD
from .utuie.net.Ushape_Trans import Generator, Discriminator
from .waternet.waternet import WaterNet
from .ugan import Generator as UGAN_G
from .ugan import Discriminator as UGAN_D
from .ultralight_vmunet import UltraLight_VM_UNet
from .uvm_net import UVM_Net
from .segnet import SegNet
from .aquatic_mamba import AquaticMambaNet


def create_network(cfg: Dict[str, Any]):
    name = cfg['name']
    if name == 'ce':
        net = ColorEnhancementNet()
    if name == 'erd':
        net = ERD(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks_res'], cfg['n_down'],
            input_h = cfg['input_h'], input_w = cfg['input_w'],
            ngf = cfg['ngf'], n_swinT = cfg['n_swinT'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            ftm_kernel_size = cfg['ftm_kernel_size'],
            ftm_padding_size = cfg['ftm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_ca = cfg['use_ca'],
            use_sa_swinT = cfg['use_sa_swinT'],
            norm_layer = cfg['norm_layer'],
            fused_window_process = cfg['fused_window_process']
        )
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
    elif name == 'ra6':
        net = RANet6(
            cfg['input_nc'], cfg['output_nc'],
            cfg['n_blocks_res'], cfg['n_down'],
            input_h = cfg['input_h'], input_w = cfg['input_w'],
            ngf = cfg['ngf'], n_swinT = cfg['n_swinT'],
            wrpm_kernel_size = cfg['wrpm_kernel_size'],
            wrpm_padding_size = cfg['wrpm_padding_size'],
            fmsm_kernel_size = cfg['fmsm_kernel_size'],
            fmsm_padding_size = cfg['fmsm_padding_size'],
            padding_type = cfg['padding_type'],
            use_dropout = cfg['use_dropout'],
            use_att_down = cfg['use_att_down'],
            use_att_up = cfg['use_att_up'],
            norm_layer = cfg['norm_layer'],
            fused_window_process = cfg['fused_window_process']
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
    elif name == 'mimo_swinT_unet6':
        net = MIMOSwinTUNet6(
            img_size = cfg['img_size'],
            num_res = cfg['num_res'],
            num_swinT = cfg['num_swinT'],
            fused_window_process=cfg['fused_window_process']
        )
    elif name == 'mimo_swinT_unet7':
        net = MIMOSwinTUNet7(
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
    elif name == 'deeplabv3':
        net = DeepLabV3(
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
    elif name == 'vmunet':
        net = VMUNet(
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            depths=cfg['depths'],
            depths_decoder=cfg['depths_decoder'],
            drop_path_rate=cfg['drop_path_rate']
        )
    elif name == 'ultralight_vmunet':
        net = UltraLight_VM_UNet(
            num_classes=cfg['num_classes'],
            input_channels=cfg['input_channels'],
            c_list=cfg['c_list'],
            split_att=cfg['split_att'],
            bridge=cfg['bridge']
        )
    elif name == 'uvm_net':
        net = UVM_Net(
            in_channels=cfg['in_channels'],
            out_channels=cfg['out_channels'],
            base_channels=cfg['base_channels'],
            bilinear=cfg['bilinear']
        )
    elif name == 'vgunet':
        net = VGUNet(
            patch_size=cfg['patch_size'],
            in_chans=cfg['in_chans'],
            out_chans=cfg['out_chans'],
            depths=cfg['depths'],
            depths_decoder=cfg['depths_decoder'],
            drop_path_rate=cfg['drop_path_rate']
        )
    elif name == 'vgunet2':
        net = VGUNet2(
            patch_size=cfg['patch_size'],
            in_chans=cfg['in_chans'],
            out_chans=cfg['out_chans'],
            depths=cfg['depths'],
            depths_decoder=cfg['depths_decoder'],
            drop_path_rate=cfg['drop_path_rate']
        )
    elif name == 'vgunet3':
        net = VGUNet3(
            patch_size=cfg['patch_size'],
            in_chans=cfg['in_chans'],
            out_chans=cfg['out_chans'],
            depths=cfg['depths'],
            depths_decoder=cfg['depths_decoder'],
            drop_path_rate=cfg['drop_path_rate']
        )
    elif name == 'vgunet4':
        net = VGUNet4(
            patch_size=cfg['patch_size'],
            in_chans=cfg['in_chans'],
            out_chans=cfg['out_chans'],
            depths=cfg['depths'],
            depths_decoder=cfg['depths_decoder'],
            drop_path_rate=cfg['drop_path_rate']
        )
    elif name == 'utuie':
        generator = Generator()
        discriminator = Discriminator()
        net = {'G': generator, 'D': discriminator}
    elif name == 'waternet':
        net = WaterNet(cfg)
    elif name == 'ugan':
        generator = UGAN_G(cfg['channels'], cfg['channels'])
        discriminator = UGAN_D(cfg['channels'])
        net = {'G': generator, 'D': discriminator}
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
    elif name == 'segnet':
        net = SegNet(
            cfg['in_chn'],
            cfg['out_chn'],
            cfg['BN_momentum']
        )
    elif name == 'aqmamba':
        net = AquaticMambaNet(
            **cfg
        )
    else:
        assert f"<{name}> is not supported!"

    return net