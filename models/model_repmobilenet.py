from typing import Optional, List, Tuple

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from misc.layer import Conv2d, FC
from torch.nn import init

def conv_dila(inp, oup, d_rate=2):

    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )
def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )
def conv2d(inp, oup):

    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class DSBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int = 1,
                 padding: int = 0,
                 dilation: int = 1,
                 groups: int = 1,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1) -> None:

        super(DSBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2d(in_channels=in_channels,
                                          out_channels=out_channels,
                                          kernel_size=kernel_size,
                                          stride=stride,
                                          padding=padding,
                                          dilation=dilation,
                                          groups=groups,
                                          bias=True)
        else:
            self.rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
                if out_channels == in_channels and stride == 1 else None

            rbr_conv = list()
            for _ in range(self.num_conv_branches):
                rbr_conv.append(self._conv_bn(kernel_size=kernel_size,
                                              padding=padding))
            self.rbr_conv = nn.ModuleList(rbr_conv)

            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1,
                                               padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def reparameterize(self):

        if self.inference_mode:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(in_channels=self.rbr_conv[0].conv.in_channels,
                                      out_channels=self.rbr_conv[0].conv.out_channels,
                                      kernel_size=self.rbr_conv[0].conv.kernel_size,
                                      stride=self.rbr_conv[0].conv.stride,
                                      padding=self.rbr_conv[0].conv.padding,
                                      dilation=self.rbr_conv[0].conv.dilation,
                                      groups=self.rbr_conv[0].conv.groups,
                                      bias=True)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias

        # Delete un-used branches
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_conv')
        self.__delattr__('rbr_scale')
        if hasattr(self, 'rbr_skip'):
            self.__delattr__('rbr_skip')

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # get weights and bias of scale branch
        kernel_scale = 0
        bias_scale = 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            kernel_scale = torch.nn.functional.pad(kernel_scale,
                                                   [pad, pad, pad, pad])

        # get weights and bias of skip branch
        kernel_identity = 0
        bias_identity = 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # get weights and bias of conv branches
        kernel_conv = 0
        bias_conv = 0
        for ix in range(self.num_conv_branches):
            _kernel, _bias = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch) -> Tuple[torch.Tensor, torch.Tensor]:

        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros((self.in_channels,
                                            input_dim,
                                            self.kernel_size,
                                            self.kernel_size),
                                           dtype=branch.weight.dtype,
                                           device=branch.weight.device)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim,
                                 self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self,
                 kernel_size: int,
                 padding: int) -> nn.Sequential:
        mod_list = nn.Sequential()
        mod_list.add_module('conv', nn.Conv2d(in_channels=self.in_channels,
                                              out_channels=self.out_channels,
                                              kernel_size=kernel_size,
                                              stride=self.stride,
                                              padding=padding,
                                              groups=self.groups,
                                              bias=False))
        mod_list.add_module('bn', nn.BatchNorm2d(num_features=self.out_channels))
        return mod_list


class Repmobilenet(nn.Module):

    def __init__(self,
                 inference_mode: bool = False,
                 num_conv_branches: int = 1) -> None:
        super().__init__()

        self.inference_mode = inference_mode
        self.in_planes = min(64, 96)
        self.num_conv_branches = num_conv_branches

        # Build stages
        self.stage0 = DSBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding=1,
                                     inference_mode=self.inference_mode)

        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(96, 2, 2,
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(192, 8, 1,
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(512, 10, 2,
                                       num_se_blocks=0)
        self.de_pred = nn.Sequential(
                                    Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu')
                                     )

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    stride_else : int) -> nn.Sequential:

        self.stride_else = stride_else
        
        if self.stride_else == 2:
            strides = [2] + [1]*(num_blocks-1)
        else:
            strides = [1]*num_blocks
        blocks = []
        for ix, stride in enumerate(strides):
            # Depthwise conv
            blocks.append(DSBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding=1,
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(DSBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding=0,
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Apply forward pass. """
        H, W = x.size(2), x.size(3)
        x1 = self.stage0(x)
        x2 = self.stage1(x1)
        x3 = self.stage2(x2)
        x4 = self.stage3(x3)
        out = self.de_pred(x4)
        out = F.interpolate(out,size = (H, W), mode='bilinear', align_corners=True) 
        return out


def Repmobilenet( inference_mode = False):
    model = Repmobilenet(inference_mode= inference_mode)
    return model


def reparameterize_model(model: torch.nn.Module) -> nn.Module:
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'reparameterize'):
            module.reparameterize()
    return model