# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
import torch.utils.checkpoint
import multigrain
from torchvision.models import resnet18, resnet50, resnet101, resnet152
from pretrainedmodels.models import senet154
from .pnasnet import pnasnet5large
from .nasnet_mobile import nasnetamobile
from collections import OrderedDict as OD
from multigrain.modules.layers import Layer

# torch.utils.checkpoint.preserve_rng_state=False


backbone_list = ['resnet18', 'resnet50', 'resnet101', 'resnet152', 'senet154', 'pnasnet5large', 'nasnetamobile']


class Features(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.base_net = net

    def forward(self, x):
        return self.base_net.features(x)


class BackBone(nn.Module):
    """
    Base networks with output dict and standarized structure
    Returns embedding, classifier_output
    """

    def __init__(self, net, **kwargs):
        super().__init__()
        if isinstance(net, str):
            # 创建基准网络
            if net not in backbone_list:
                raise ValueError('Available backbones:', ', '.join(backbone_list))
            net = multigrain.backbones.backbone.__dict__[net](**kwargs)

        # 解析网络各个子模块，划分为特征提取层、池化层、分类器层
        children = list(net.named_children())
        self.pre_classifier = None
        if type(net).__name__ == 'ResNet':
            self.features = nn.Sequential(OD(children[:-2]))
            self.pool = children[-2][1]
            self.classifier = children[-1][1]
        elif type(net).__name__ == 'SENet':
            self.features = nn.Sequential(OD(children[:-3]))
            self.pool = children[-3][1]
            self.pre_classifier = children[-2][1]
            self.classifier = children[-1][1]
        elif type(net).__name__ in ['PNASNet5Large', 'NASNetAMobile']:
            self.features = nn.Sequential(Features(net), nn.ReLU())
            self.pool = children[-3][1]
            self.pre_classifier = children[-2][1]
            self.classifier = children[-1][1]
        else:
            raise NotImplementedError('Unknown base net', type(net).__name__)
        self.whitening = None

    def forward(self, input):
        output = {}
        if isinstance(input, list):
            # 如果input是列表格式，那么表明输入数据大小可能不一致
            # 可能存在不同的输入大小，逐个提取特征
            # for lists of tensors of unequal input size
            features = map(self.features, [i.unsqueeze(0) for i in input])
            # 逐个特征进行池化操作，最后连接在一起
            embedding = torch.cat([self.pool(f) for f in features], 0)
        else:
            # 针对相同大小的输入数据，进行批运算
            features = self.features(input)
            embedding = self.pool(features)
        if self.whitening is not None:
            # 如果白化操作不为空，那么执行白化操作
            embedding = self.whitening(embedding)

        # 计算分类输出
        classifier_input = embedding
        if self.pre_classifier is not None:
            classifier_input = self.pre_classifier(classifier_input)
        classifier_output = self.classifier(classifier_input)

        # 返回特征嵌入向量和分类输出结果
        return embedding, classifier_output
