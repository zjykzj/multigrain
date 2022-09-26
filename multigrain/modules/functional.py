# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import math
from torch.nn import functional as F


def add_bias_channel(x, dim=1):
    """
    增加偏置通道，原先大小为[N, K], 赋值成[N, 1]大小，初始化为1
    最后按照dim=1维度进行连接，得到[N, K+1]
    """
    one_size = list(x.size())
    one_size[dim] = 1
    one = x.new_ones(one_size)
    return torch.cat((x, one), dim)


def flatten(x, keepdims=False):
    """
    [B, C, H, W] -> [B, C*H*W]
    如果保持原先的维度，那么设置为[1, 1, B, C*H*W]
    Flattens B C H W input to B C*H*W output, optionally retains trailing dimensions.
    """
    y = x.view(x.size(0), -1)
    if keepdims:
        for d in range(y.dim(), x.dim()):
            y = y.unsqueeze(-1)
    return y


def gem(x, p=3, eps=1e-6, clamp=True, add_bias=False, keepdims=False):
    """
    GeM池化操作
    如果池化因子p设置为无穷大，那么近似于最大池化操作
    如果池化因子p设置为1, 那么近似于平均池化操作

    如果设置了截断选项，首先对输入进行精度截断操作
    对输入执行幂运算，幂为p
    然后执行平均池化操作
    最后执行幂运算，幂为1.0/p

    如果设置了偏置选项，那么添加偏置通道
    如果不需要保持原先维度大小，那么将输出拉平到2D
    """
    if p == math.inf or p is 'inf':
        x = F.max_pool2d(x, (x.size(-2), x.size(-1)))
    elif p == 1 and not (torch.is_tensor(p) and p.requires_grad):
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
    else:
        if clamp:
            x = x.clamp(min=eps)
        x = F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
    if add_bias:
        x = add_bias_channel(x)
    if not keepdims:
        x = flatten(x)
    return x


def apply_pca(vt, pca_P=None, pca_m=None):
    """
    白化操作近似于旋转矩阵+Shift(偏移)操作
    """
    do_rotation = torch.is_tensor(pca_P) and pca_P.numel() > 0
    do_shift = torch.is_tensor(pca_P) and pca_P.numel() > 0

    if do_rotation or do_shift:
        if do_shift:
            # 偏移操作
            vt = vt - pca_m
        if do_rotation:
            # 矩阵乘法，执行旋转操作
            vt = torch.matmul(vt, pca_P)
    return vt


def l2n(x, eps=1e-6, dim=1):
    """
    基于指定维度计算L2范数，执行归一化操作
    """
    x = x / (torch.norm(x, p=2, dim=dim, keepdim=True) + eps).expand_as(x)
    return x
