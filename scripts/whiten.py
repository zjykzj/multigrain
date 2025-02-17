# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import DataLoader, Subset

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import ListDataset
from multigrain import utils
from multigrain.backbones import backbone_list
from multigrain.lib.whiten import get_whiten

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os.path as osp

# 计时
tic, toc = utils.Tictoc()

"""

目标：

"""


def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)

    argfile = osp.join(osp.join(args.expdir), 'whiten_args.yaml')

    args.cuda = not args.no_cuda

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    collate_fn = dict(collate_fn=list_collate) if args.input_crop == 'rect' else {}

    # 加载验证集预处理器
    transforms = get_transforms(input_size=args.input_size, crop=(args.input_crop == 'square'), need=('val',),
                                backbone=args.backbone)
    # 给定白化文件路径以及白化数据根路径, 创建数据类
    dataset = ListDataset(args.whiten_path, args.whiten_list, transforms['val'])
    # 如果指定了参与白化训练的图像数据, 则创建子数据集
    if args.num_whiten_images != -1:
        dataset = Subset(dataset, list(range(args.num_whiten_images)))
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True, **collate_fn)

    # 创建MultiGrain模型
    # 关键一：不需要在MultiGrain模型内部计算嵌入特征对应的正负样本对
    model = get_multigrain(args.backbone, include_sampling=False, pretrained_backbone=args.pretrained_backbone)

    if args.cuda:
        model = utils.cuda(model)

    p = model.pool.p

    # 加载预训练权重
    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        resume_epoch = checkpoints.resume(model, resume_epoch=args.resume_epoch,
                                          resume_from=args.resume_from, return_extra=False)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    # 是否使用指定的池化因子p
    if args.pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    print(model)

    # 关键二：初始化模型白化设置
    model.init_whitening()
    # 关键三：设置模型评估模式
    model.eval()

    print("Computing embeddings...")
    # 计算嵌入特征
    embeddings = []
    for i, batch in enumerate(loader):
        if i % (len(loader) / 100) < 1:
            print("{}/{} ({}%)".format(i, len(loader), int(i // (len(loader) / 100))))
        with torch.no_grad():
            if args.cuda:
                batch = utils.cuda(batch)
            embeddings.append(model(batch)['embedding'].cpu())
    #
    embeddings = torch.cat(embeddings)
    if args.no_include_last:
        # 卷积层偏移对白化计算有影响吗？
        # 参数默认设置为否，应该没啥影响
        embeddings = embeddings[:, :-1]

    print("Computing whitening...")
    # 得到了嵌入向量列表，可以计算白化参数
    m, P = get_whiten(embeddings)

    if args.no_include_last:
        # add an preserved channel to the PCA
        # 作用于全连接层，默认不使用
        m = torch.cat((m, torch.tensor([0.0])), 0)
        D = P.size(0)
        P = torch.cat((P, torch.zeros(1, D)), 0)
        P = torch.cat((P, torch.cat((torch.zeros(D, 1), torch.tensor([1.0])), 1)), 1)

    # 集成白化参数
    model.integrate_whitening(m, P)

    if not args.dry:
        checkpoints.save(model, resume_epoch if resume_epoch != -1 else 0)


if __name__ == "__main__":
    # 针对MultiGrain计算白化矩阵。使用全连接层+shift操作模拟PCA白化操作
    parser = ArgumentParser(description="Whitening computation for MultiGrain model, computes the whitening matrix",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # 到处训练权重路径
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500_whitened',
                        help='destination directory for checkpoint')
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    parser.add_argument('--resume-from', default=None, help='source experiment to whiten')
    # 输入大小
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    # 是否执行中心裁剪，默认为rect，表示不进行裁剪，直接缩放
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    # 批量训练数目
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    # 主干网络
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    # 预训练权重
    parser.add_argument('--pretrained-backbone', action='store_true', help='use pretrained backbone')
    # 是否手动设置池化因子
    parser.add_argument('--pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    # 是否移除全连接层的bias通道，默认为否
    parser.add_argument('--no-include-last', action='store_true',
                        help='remove last channel from PCA (useful to not include "bias multiplier" channel)')
    # 计算白化矩阵的图片列表文件
    parser.add_argument('--whiten-list', default='data/whiten.txt', help='list of images to compute whitening')
    # 计算白化矩阵的图像文件根路径
    parser.add_argument('--whiten-path', default='data/whiten', help='whitening data root')
    # 参与白化训练的数据, 默认为-1(使用全部数据)
    parser.add_argument('--num-whiten-images', default=-1, type=int,
                        help='number of images used in whitening. (-1 -> all in list)')
    # 数据加载器线程数
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()

    run(args)
