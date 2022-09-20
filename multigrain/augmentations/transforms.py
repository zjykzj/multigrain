# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torchvision.transforms.functional as F
from torchvision import transforms

from multigrain.datasets import IN1K
from .autoaugment import ImageNetPolicy


class Resize(transforms.Resize):
    """
    Resize with a ``largest=False'' argument
    allowing to resize to a common largest side without cropping

    图像缩放，较大边长缩放到指定size大小
    """

    def __init__(self, size, largest=False, **kwargs):
        super().__init__(size, **kwargs)
        self.largest = largest

    @staticmethod
    def target_size(w, h, size, largest=False):
        if (h < w) == largest:
            # 如果宽大于高，那么宽设置为size，高按照等比例缩放
            w, h = size, int(size * h / w)
        else:
            # 如果宽小于高，那么高设置为size，宽按照等比例缩放
            w, h = int(size * w / h), size
        size = (h, w)
        return size

    def __call__(self, img):
        size = self.size
        w, h = img.size
        target_size = self.target_size(w, h, size, self.largest)
        return F.resize(img, target_size, self.interpolation)

    def __repr__(self):
        r = super().__repr__()
        return r[:-1] + ', largest={})'.format(self.largest)


class Lighting(object):
    """
    PCA jitter transform on tensors

    See https://zhuanlan.zhihu.com/p/69439309
    PCA抖动：首先按照RGB三个颜色通道计算均值和标准差，再在整个训练集上计算协方差矩阵，进行特征分解，得到特征向量和特征值，用来做PCA Jittering
    """

    def __init__(self, alpha_std, eig_val, eig_vec):
        self.alpha_std = alpha_std
        self.eig_val = torch.as_tensor(eig_val, dtype=torch.float).view(1, 3)
        self.eig_vec = torch.as_tensor(eig_vec, dtype=torch.float)

    def __call__(self, data):
        if self.alpha_std == 0:
            return data
        alpha = torch.empty(1, 3).normal_(0, self.alpha_std)
        rgb = ((self.eig_vec * alpha) * self.eig_val).sum(1)
        data += rgb.view(3, 1, 1)
        data /= 1. + self.alpha_std
        return data


class Bound(object):
    """
    设置数值最大、最小值，截断精度
    """

    def __init__(self, lower=0., upper=1.):
        self.lower = lower
        self.upper = upper

    def __call__(self, data):
        return data.clamp_(self.lower, self.upper)


def get_transforms(Dataset=IN1K, input_size=224, kind='full', crop=True, need=('train', 'val'), backbone=None):
    """
    @param Dataset: 数据集类型
    @param input_size: 输入大小
    @param kind: 预处理列表类型 论文设置了多种预处理列表，默认为full
    @param crop: 是否在验证集预处理阶段进行中央裁剪
    @param need: 针对不同阶段(训练或者验证阶段)进行的数据处理
    @param backbone: 针对指定的主干网络使用不同的均值/方差

    从实现上看，不管是训练还是测试阶段，预处理器都进行了固定大小缩放操作，保证同一批图片都是相同大小的
    """
    mean, std = Dataset.MEAN, Dataset.STD
    if backbone is not None and backbone in ['pnasnet5large', 'nasnetamobile']:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transformations = {}
    if 'train' in need:
        if kind == 'torch':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        elif kind == 'full':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3),
                transforms.ToTensor(),
                Lighting(0.1, Dataset.EIG_VALS, Dataset.EIG_VECS),
                Bound(0., 1.),
                transforms.Normalize(mean, std),
            ])
        elif kind == 'senet':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                Lighting(0.1, Dataset.EIG_VALS, Dataset.EIG_VECS),
                Bound(0., 1.),
                transforms.Normalize(mean, std),
            ])
        elif kind == 'AA':
            transformations['train'] = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            raise ValueError('Transforms kind {} unknown'.format(kind))
    if 'val' in need:
        if crop:
            transformations['val'] = transforms.Compose(
                [Resize(int((256 / 224) * input_size)),  # to maintain same ratio w.r.t. 224 images
                 transforms.CenterCrop(input_size),
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transformations['val'] = transforms.Compose(
                [Resize(input_size, largest=True),  # to maintain same ratio w.r.t. 224 images
                 transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
    return transformations


transforms_list = ['torch', 'full', 'senet', 'AA']
