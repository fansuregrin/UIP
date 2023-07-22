import torch.nn.init as t_init
import torch.nn as nn


def init_weight_bias(m):
    if isinstance(m, nn.Conv2d):
        t_init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            t_init.constant_(m.bias, 0.0)