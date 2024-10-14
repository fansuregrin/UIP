"""UVM-Net
code: https://github.com/zzr-idam/UVM-Net

@article{zheng2024u,
  title={U-shaped Vision Mamba for Single Image Dehazing},
  author={Zheng, Zhuoran and Wu, Chen},
  journal={arXiv preprint arXiv:2402.04139},
  year={2024}
}
"""
from networks import NetworkCreator, network_creators
from .uvm_net import UVM_Net

__all__ = ['UVM_Net']


@network_creators.register('uvm_net')
class UVMNetCreator(NetworkCreator):
    def __init__(self):
        super().__init__()

    def create_network(cfg):
        return UVM_Net(**cfg)