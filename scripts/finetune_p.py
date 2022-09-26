# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import SGD, lr_scheduler

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import IN1K, IdDataset
from multigrain import utils
from multigrain.modules import MultiOptim
from multigrain.backbones import backbone_list

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml
import os.path as osp
from collections import defaultdict, Counter
from collections import OrderedDict as OD

# 计时
tic, toc = utils.Tictoc()

"""
微调训练池化算子GeM的超参数p

关于数据：

1. 从ImageNet训练集中采集部分数据（每类采样50张 - args.images_per_class）
2. 仍旧使用ImageNet验证集进行评估

关于模型：

1. 设置超参数p可学习
2. 设置eval状态

关于损失函数：

1. 交叉熵损失

关于优化器：

1. SGD+Momentum
2. 仅更新GeM超参数p

因为是微调操作，所以设置低学习率进行操作

另外，微调训练的目的是为了搭配图像输入和超参数p的使用，所以不同的输入大小需要微调不一样的p

"""


def run(args):
    # 序列化输入参数
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)
    argfile = osp.join(osp.join(args.expdir), 'finetune_p_args.yaml')

    if osp.isfile(argfile):
        oldargs = yaml.load(open(argfile))
        if oldargs != args.__dict__:
            print('WARNING: Changed configuration keys compared to stored experiment')
            utils.arguments.compare_dicts(oldargs, args.__dict__, verbose=True)

    args.cuda = not args.no_cuda
    args.validate_first = not args.no_validate_first
    args.validate = not args.no_validate

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    # 创建预处理器 仅需操作验证集预处理器
    transforms = get_transforms(IN1K, args.input_size, crop=(args.input_crop == 'square'), need=('val',),
                                backbone=args.backbone)
    # 创建训练集和验证集数据类
    datas = {}
    for split in ('train', 'val'):
        datas[split] = IdDataset(IN1K(args.imagenet_path, split, transform=transforms['val']))
    # 从训练集中采集每类微调数据, 数据量不超过指定数目
    loaders = {}
    collate_fn = dict(collate_fn=list_collate) if args.input_crop == 'rect' else {}
    selected = []
    count = Counter()
    for i, label in enumerate(datas['train'].dataset.labels):
        if count[label] < args.images_per_class:
            selected.append(i)
            count[label] += 1
    # 创建子数据集类
    datas['train'].dataset = Subset(datas['train'].dataset, selected)
    # 创建数据加载器
    loaders['train'] = DataLoader(datas['train'], batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.workers, pin_memory=True, **collate_fn)
    loaders['val'] = DataLoader(datas['val'], batch_size=args.batch_size, shuffle=args.shuffle_val,
                                num_workers=args.workers, pin_memory=True, **collate_fn)

    # 获取MultiGrain模型, 设置GeM超参数p可学习
    model = get_multigrain(args.backbone, include_sampling=False,
                           pretrained_backbone=args.pretrained_backbone, learn_p=True)

    # 创建损失函数, 使用交叉熵损失
    criterion = torch.nn.CrossEntropyLoss()
    if args.cuda:
        criterion = utils.cuda(criterion)
        model = utils.cuda(model)

    # 创建优化器, 仅梯度更新GeM超参数p
    optimizers = OD()
    p = model.pool.p
    optimizers['p'] = SGD([p], lr=args.learning_rate, momentum=args.momentum)
    optimizers = MultiOptim(optimizers)

    def training_step(batch):
        optimizers.zero_grad()

        output_dict = model(batch['input'])
        loss = criterion(output_dict['classifier_output'], batch['classifier_target'])
        top1, top5 = utils.accuracy(output_dict['classifier_output'].data, batch['classifier_target'].data, topk=(1, 5))

        p.grad = torch.autograd.grad(loss, p)[0]  # partial backward
        optimizers.step()

        return OD([
            ('cross_entropy', loss.item()),
            ('p', p.item()),
            ('top1', top1),
            ('top5', top5),
        ])

    def validation_step(batch):
        with torch.no_grad():
            output_dict = model(batch['input'])
            target = batch['classifier_target']
            xloss = criterion(output_dict['classifier_output'], target)
            top1, top5 = utils.accuracy(output_dict['classifier_output'], target, topk=(1, 5))

        return OD([
            ('cross_entropy', xloss.item()),
            ('top1', top1),
            ('top5', top5),
        ])

    metrics_history = OD()

    # 加载预训练权重
    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        epoch = checkpoints.resume(model, metrics_history=metrics_history,
                                   resume_epoch=args.resume_epoch, resume_from=args.resume_from)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    # 是否初始化池化因子
    if args.init_pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.init_pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    print(model)

    def loop(loader, step, epoch, prefix=''):  # Training or validation loop
        metrics = defaultdict(utils.HistoryMeter if prefix == 'train_' else utils.AverageMeter)
        # 计时
        tic()
        # 批量数据加载器
        for i, batch in enumerate(loader):
            # 对于训练阶段, 进行学习率衰减
            if prefix == 'train_':
                lr = args.learning_rate * (1 - i / len(loader)) ** args.learning_rate_decay_power
                optimizers['p'].param_groups[0]['lr'] = lr
            if args.cuda:
                batch = utils.cuda(batch)
            # 计算批量数据预处理时间
            data_time = 1000 * toc()
            # 重新计时
            tic()
            step_metrics = step(batch)
            step_metrics['data_time'] = data_time
            # 计算模型推理时间(训练或者验证)
            step_metrics['batch_time'] = 1000 * toc()
            # 重新计时
            tic()
            for (k, v) in step_metrics.items():
                metrics[prefix + k].update(v, len(batch['input']))
            print(logging.str_metrics(metrics, iter=i, num_iters=len(loader), epoch=epoch, num_epochs=epoch))
        print(logging.str_metrics(metrics, epoch=epoch, num_epochs=epoch))
        # 结束计时
        toc()
        if prefix == 'val_':
            return OD((k, v.avg) for (k, v) in metrics.items())
        return OD((k, v.hist) for (k, v) in metrics.items())

    # 首先进行验证集评估
    if args.validate_first and 0 not in metrics_history:
        model.eval()
        metrics_history[epoch] = loop(loaders['val'], validation_step, epoch, 'val_')
        checkpoints.save_metrics(metrics_history)

    # 然后训练超参数p
    model.eval()  # freeze batch normalization
    metrics = loop(loaders['train'], training_step, epoch, 'train_')
    metrics['last_p'] = p.item()

    # 完成训练后, 再次进行验证集验证
    if args.validate:
        model.eval()
        metrics.update(loop(loaders['val'], validation_step, epoch + 1, 'val_'))

        metrics_history[epoch + 1] = metrics

    if not args.dry:
        utils.make_plots(metrics_history, args.expdir)
        checkpoints.save(model, epoch + 1, optimizers, metrics_history)


if __name__ == "__main__":
    # 通过微调训练确定不同输入大小对应的指数p
    parser = ArgumentParser(description="""GeM p exponent finetuning script for MultiGrain model, 
                                           computes the p exponent for a given input size by fine-tuning""",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # 输出目录
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500', help='experiment destination directory')
    # 是否需要打乱验证数据集, 默认为否
    parser.add_argument('--shuffle-val', action='store_true', help='shuffle val. dataset')
    # 恢复训练轮数, 默认从头开始
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    # 预训练权重路径
    parser.add_argument('--resume-from', default=None, help='resume checkpoint file/folder')
    # 最初学习率
    parser.add_argument('--learning-rate', default=0.005, type=float, help='base learning rate')
    # 学习率衰减指数
    parser.add_argument('--learning-rate-decay-power', default=0.9, type=float,
                        help='Power in polynomial learning rate decay')
    # SGD动量
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum in SGD')
    # 图像输入大小, 默认为500
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    # 是否裁剪图像, 应该是说是否执行中央裁剪
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    # 批量输入大小
    parser.add_argument('--batch-size', default=4, type=int, help='batch size')
    # 指定每类图像数目
    parser.add_argument('--images-per-class', default=50, type=int,
                        help='use a training subset of N images per class for the finetuning')
    # 主干网络
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    # 预训练网络权重
    parser.add_argument('--pretrained-backbone', action='store_const', const='imagenet', help='use pretrained backbone')
    # 是否在训练之前执行验证, 默认为否(先执行验证)
    parser.add_argument('--no-validate-first', action='store_true', help='do not validate before training')
    # 应该和上面一样, 不知道为什么
    parser.add_argument('--no-validate', action='store_true',
                        help='do not validate after training')
    # 训练最开始的池化指数, 默认使用预训练模型中的池化指数
    parser.add_argument('--init-pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    # 是否在GPU环境下操作, 默认为是(在GPU环境下操作)
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    # ImageNet数据集根路径
    parser.add_argument('--imagenet-path', default='data/ilsvrc2012', help='ImageNet data root')
    # 一些ImageNet配置文件, 保存成文件可以加速读取操作
    parser.add_argument('--preload-dir-imagenet', default=None,
                        help='preload imagenet in this directory (useful for slow networks')
    # 数据加载器操作线程
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    # 是否需要保存日志
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()

    run(args)
