# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from sklearn.decomposition import PCA
import torch
import numpy as np


def get_whiten(X):
    # X: [N, D]
    # 创建PCA对象，打开白化开关，设置转换后维度为D
    # 使用sklearn库计算PCA白化矩阵
    pca = PCA(whiten=True, n_components=X.size(1))
    pca.fit(X.detach().cpu().numpy())
    m = torch.tensor(pca.mean_, dtype=torch.float)
    P = torch.tensor(pca.components_.T / np.sqrt(pca.explained_variance_), dtype=torch.float)

    # 返回的是均值向量（偏移操作）和特征矩阵（全连接层操作）
    return m, P
