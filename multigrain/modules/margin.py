# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch import nn
import torch
import numpy as np


class DistanceWeightedSampling(nn.Module):
    r"""Distance weighted sampling.
    See "sampling matters in deep embedding learning" paper for details.
    Implementation similar to https://github.com/chaoyuaw/sampling_matters
    """

    def __init__(self, cutoff=0.5, nonzero_loss_cutoff=1.4):
        super().__init__()
        self.cutoff = cutoff
        self.nonzero_loss_cutoff = nonzero_loss_cutoff

    @staticmethod
    def get_distance(x):
        """
        Helper function for margin-based loss. Return a distance matrix given a matrix.
        Returns 1 on the diagonal (prevents numerical errors)
        """
        # 获取特征列表长度
        n = x.size(0)
        # 计算每个特征向量的平方和 [N, D] -> [N, 1]
        square = torch.sum(x ** 2.0, dim=1, keepdim=True)
        # 计算向量之间的欧式距离
        # square + square.t(): [N, 1] + [1, N] -> [N, N]
        # torch.matmul(x, x.t()): [N, D] * [D, N] -> [N, N]
        # 等价于 x1**2 + x2**2 - 2*x1*x2 = (x1 - x2)**2
        # 这种情况下，矩阵对角特征值为0，因为x1 == x2
        distance_square = square + square.t() - (2.0 * torch.matmul(x, x.t()))
        # 开根号，计算欧式距离
        # 对角特征设置为1, 避免数值计算错误
        return torch.sqrt(distance_square + torch.eye(n, dtype=x.dtype, device=x.device))

    def forward(self, embedding, target):
        """
        embedding: [N, D]
        target: [N]

        Inputs:
            - embedding: embeddings of images in batch
            - target: id of instance targets

        Outputs:
            - a dict with
               * 'anchor_embeddings'
               * 'negative_embeddings'
               * 'positive_embeddings'
               with sampled embeddings corresponding to anchors, negatives, positives
        """

        # 获取特征向量的个数和维度
        B, C = embedding.size()[:2]
        embedding = embedding.view(B, C)

        # 计算特征向量之间的欧式距离
        distance = self.get_distance(embedding)
        # 精度截断
        distance = torch.clamp(distance, min=self.cutoff)

        # Subtract max(log(distance)) for stability.
        log_weights = ((2.0 - float(C)) * torch.log(distance)
                       - (float(C - 3) / 2) * torch.log(1.0 - 0.25 * (distance ** 2.0)))
        # 计算指数权重
        weights = torch.exp(log_weights - log_weights.max())

        # [N] -> [1, N]
        unequal = target.view(-1, 1)
        # 计算不相同target的下标
        unequal = (unequal != unequal.t())

        # 计算掩码，过滤对角特征以及距离小于指定阈值的特征
        weights = weights * (unequal & (distance < self.nonzero_loss_cutoff)).float()
        # 剩余特征进行归一化
        weights = weights / torch.sum(weights, dim=1, keepdim=True)

        a_indices = []
        p_indices = []
        n_indices = []

        np_weights = weights.detach().cpu().numpy()
        unequal_np = unequal.cpu().numpy()

        # 遍历每条特征
        for i in range(B):
            # 计算和特征i拥有相同标签的特征的下标
            # 这里面包括了下标i
            same = (1 - unequal_np[i]).nonzero()[0]

            # 如果权重求和不为空
            # 注意： 仅不相同标签的特征下标才会出现权重值
            if np.isnan(np_weights[i].sum()):  # 0 samples within cutoff, sample uniformly
                np_weights_ = unequal_np[i].astype(float)
                # 权重归一化
                np_weights_ /= np_weights_.sum()
            else:
                # 上面已经归一化了
                np_weights_ = np_weights[i]

            # 采样负样本列表，长度和正样本列表一致
            try:
                # 基于采样概率进行，对于概率为0的情况，其采集可能性为0
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_, replace=False).tolist()
            except ValueError:  # cannot always sample without replacement
                # 如果负样本长度小于正样本长度，那么允许重复采样
                n_indices += np.random.choice(B, len(same) - 1, p=np_weights_).tolist()

            for j in same:
                if j != i:
                    # 采集正样本对
                    a_indices.append(i)
                    p_indices.append(j)

        return {'anchor_embeddings': embedding[a_indices],
                'negative_embeddings': embedding[n_indices],
                'positive_embeddings': embedding[p_indices]}


class MarginLoss(nn.Module):
    r"""Margin based loss.

    Parameters
    ----------
    beta_init: float
        Initial beta
    margin : float
        Margin between positive and negative pairs.
    """

    def __init__(self, beta_init=1.2, margin=0.2):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self._margin = margin

    def forward(self, anchor_embeddings, negative_embeddings, positive_embeddings, eps=1e-8):
        """

        Inputs:
            - input_dict: 'anchor_embeddings', 'negative_embeddings', 'positive_embeddings'

        Outputs:
            - Loss.
        """

        # 计算正样本对之间的欧式距离
        d_ap = torch.sqrt(torch.sum((positive_embeddings - anchor_embeddings) ** 2, dim=1) + eps)
        # 计算负样本对之间的欧式距离
        d_an = torch.sqrt(torch.sum((negative_embeddings - anchor_embeddings) ** 2, dim=1) + eps)

        # 计算正样本对损失
        pos_loss = torch.clamp(d_ap - self.beta + self._margin, min=0.0)
        # 计算负样本对损失
        neg_loss = torch.clamp(self.beta - d_an + self._margin, min=0.0)

        # 计算符合条件的数目
        pair_cnt = float(torch.sum((pos_loss > 0.0) + (neg_loss > 0.0)).item())

        # 归一化操作
        # Normalize based on the number of pairs
        loss = (torch.sum(pos_loss + neg_loss)) / max(pair_cnt, 1.0)

        return loss


class SampledMarginLoss(nn.Module):
    """
    Combines DistanceWeightedSampling + Margin Loss
    """

    def __init__(self, sampling_args={}, margin_args={}):
        super().__init__()
        self.sampling = DistanceWeightedSampling(**sampling_args)
        self.margin = MarginLoss(**margin_args)

    def forward(self, embedding, target):
        sampled_dict = self.sampling(embedding, target)
        loss = self.margin(**sampled_dict)
        return loss
