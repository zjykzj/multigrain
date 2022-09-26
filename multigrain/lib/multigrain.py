# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch import nn
from multigrain.backbones import BackBone
import torch.utils.model_zoo as model_zoo
from multigrain.modules.layers import Layer, Select
from multigrain.modules import DistanceWeightedSampling
from collections import OrderedDict as OD

__all__ = ['multigrain']

model_urls = {
    ('multigrain_resnet50'): '',
}

"""
将完整的分类网络拆分为u几个模块：

1. 主干Backbone
2. 池化层
3. 分类器

替换掉池化层，使用GeM池化

返回两部分内容：

1. 特征向量
2. 分类输出
"""


class MultiGrain(BackBone):
    """
    创建Multigrain架构
    Implement MultiGrain by changing the pooling layer of the backbone into GeM pooling with exponent p,
    and adding DistanceWeightedSampling for the margin loss.
    """

    def __init__(self, backbone, p=3.0, include_sampling=True, learn_p=False, **kwargs):
        super().__init__(backbone, **kwargs)
        if not torch.is_tensor(p):
            p = torch.tensor(p)
            if learn_p:
                p.requires_grad = True
        self.pool = Layer('gem', p=p)
        self.normalize = Layer('l2n')
        if include_sampling:
            # 对于分类任务而言，不需要执行距离加权采样
            # 对于检索任务而言，需要从分类网络计算得到的特征向量和分类输出的基础上进一步进行采样，获取正负样本对特征向量
            self.weighted_sampling = DistanceWeightedSampling()
        self.whitening = None

    def load_state_dict(self, D, *args, **kwargs):
        # adjust whitening and bias during load_state_dict
        for (k, v) in D.items():
            parts = k.split('.')
            if parts[-1] in ('pca_P', 'pca_m'):
                if self.whitening is None:
                    self.init_whitening(loading_checkpoint=True)
                getattr(self.whitening.pca, parts[-1]).resize_(v.size())
        super().load_state_dict(D, *args, **kwargs)

    def init_whitening(self, loading_checkpoint=False):
        """
        Initialize whitening operation (see scripts/whiten.py)
        """
        self.whitening = nn.Sequential(OD(normalize=Layer('l2n'),
                                          pca=Layer('apply_pca', pca_P=torch.tensor([]), pca_m=torch.tensor([]))))
        # integrate bias in classifier to make it invariant to the input normalization
        self.pool.kwargs['add_bias'] = True
        if self.pre_classifier is not None:
            self.pre_classifier = Select(self.pre_classifier, -1)
        W, b = self.classifier.weight, self.classifier.bias
        W.data = torch.cat((W.data, b.data.view(-1, 1)), 1)
        if not loading_checkpoint:
            self.classifier.bias = None

    def integrate_whitening(self, m, P):
        """
        Set whitening parameters and add their reverse in classifier (see scripts/whiten.py)
        """
        Pinv = P.t().double().inverse()
        if False:
            # does not work w/ torch 1.2
            self.whitening.pca.pca_m.data.resize_(m.size()).copy_(m)
            self.whitening.pca.pca_P.data.resize_(P.size()).copy_(P)
        else:
            self.whitening.pca.pca_m = torch.clone(m)
            self.whitening.pca.pca_P = torch.clone(P)
        # 白化操作仅作用于检索任务，所以对于全连接层而言，需要通过反向矩阵消除影响
        # 当然这个工程的操作比较高阶哈，我的话就直接拷贝两份主干特征，一份用于检索特征的白化操作，另一份用于分类特征的分类输出
        W = self.classifier.weight
        self.classifier.bias = nn.Parameter(m.to(W.device).matmul(W.data.t()))
        W.data = torch.matmul(W.data.double(), Pinv.to(W.device)).float()

    def forward(self, input, instance_target=None, **kwargs):
        if isinstance(instance_target, list):
            instance_target = torch.stack(instance_target)
        output_dict = {'instance_target': instance_target}
        output_dict['embedding'], output_dict['classifier_output'] = super().forward(input, **kwargs)
        output_dict['normalized_embedding'] = self.normalize(output_dict['embedding'])

        if hasattr(self, 'weighted_sampling') and instance_target is not None:
            sampled = self.weighted_sampling(output_dict['normalized_embedding'], instance_target)
            output_dict.update(sampled)

        return output_dict


def get_multigrain(backbone='resnet50', pretrained=None, pretrained_backbone=None, **kwargs):
    kwargs['pretrained'] = pretrained_backbone
    model = MultiGrain(backbone, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['multigrain_' + backbone]))
    return model
