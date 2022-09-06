import torch
from torch.utils.data import DataLoader
import faiss

import set_path
from multigrain.utils import logging
from multigrain.augmentations import get_transforms
from multigrain.lib import get_multigrain, list_collate
from multigrain.datasets import IN1K, IdDataset
from multigrain import utils
from multigrain.backbones import backbone_list

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import OrderedDict as OD
from collections import defaultdict
import yaml
import os.path as osp

# 计时
tic, toc = utils.Tictoc()


def run(args):
    argstr = yaml.dump(args.__dict__, default_flow_style=False)
    print('arguments:')
    print(argstr)

    argfile = osp.join(osp.join(args.expdir), 'evaluate_args.yaml')

    args.cuda = not args.no_cuda

    if not args.dry:
        utils.ifmakedirs(args.expdir)
        logging.print_file(argstr, argfile)

    # 自定义批量数据分配函数
    collate_fn = dict(collate_fn=list_collate) if args.input_crop == 'rect' else {}
    # 创建预处理器
    transforms = get_transforms(
        input_size=args.input_size,
        crop=(args.input_crop == 'square'),
        need=('val',),
        backbone=args.backbone
    )

    # 当前仅支持ImageNet数据类
    if args.dataset.startswith('imagenet'):
        dataset = IdDataset(
            IN1K(
                args.imagenet_path,
                args.dataset[len('imagenet-'):],
                transform=transforms['val']
            )
        )
        # 进行分类任务评估
        mode = "classification"
    else:
        raise NotImplementedError(
            "Retrieval evaluations not implemented yet, check datasets/retrieval.py to implement the evaluations.")

    # 将数据类载入数据加载器
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=args.shuffle,
                        pin_memory=True, **collate_fn)

    # 创建模型
    model = get_multigrain(args.backbone, include_sampling=False, pretrained_backbone=args.pretrained_backbone)

    p = model.pool.p
    # 加载预训练权重
    checkpoints = utils.CheckpointHandler(args.expdir)

    if checkpoints.exists(args.resume_epoch, args.resume_from):
        epoch = checkpoints.resume(model, resume_epoch=args.resume_epoch, resume_from=args.resume_from,
                                   return_extra=False)
    else:
        raise ValueError('Checkpoint ' + args.resume_from + ' not found')

    # 重载池化因子
    if args.pooling_exponent is not None:  # overwrite stored pooling exponent
        p.data.fill_(args.pooling_exponent)

    print("Multigrain model with {} backbone and p={} pooling:".format(args.backbone, p.item()))
    print(model)

    # 是否在GPU环境下操作
    if args.cuda:
        model = utils.cuda(model)

    model.eval()  # freeze batch normalization

    print("Evaluating", args.dataset)

    metrics_history = OD()
    metrics = defaultdict(utils.HistoryMeter)

    embeddings = []
    index = None
    # 计时开始
    tic()
    # 遍历数据加载器
    for i, batch in enumerate(loader):
        with torch.no_grad():
            if args.cuda:
                batch = utils.cuda(batch)
            # 计算数据加载时间
            metrics["data_time"].update(1000 * toc())
            # 重新计时
            tic()
            # 模型推理
            output_dict = model(batch['input'])
        if mode == "classification":
            # 获取分类数据集的数据标签
            target = batch['classifier_target']
            # 计算top1/top5
            top1, top5 = utils.accuracy(output_dict['classifier_output'], target, topk=(1, 5))
            # 记录相关的结果
            metrics["val_top1"].update(top1, n=len(batch['input']))
            metrics["val_top5"].update(top5, n=len(batch['input']))
        elif mode == "retrieval":
            # 进行检索任务评估
            if index is None:
                # 创建faiss评估器
                index = faiss.IndexFlatL2(descriptors.size(1))
            descriptors = output_dict['normalized_embedding']
            for e in descriptors.cpu():
                index.append(e)
        # 记录批量推理和评估时间
        metrics["batch_time"].update(1000 * toc())
        # 重新计时
        tic()
        print(logging.str_metrics(metrics, iter=i, num_iters=len(loader), epoch=epoch, num_epochs=epoch))
    print(logging.str_metrics(metrics, epoch=epoch, num_epochs=1))
    for k in metrics:
        metrics[k] = metrics[k].avg
    toc()

    metrics_history[epoch] = metrics
    checkpoints.save_metrics(metrics_history)


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # 数据集
    parser.add_argument('--dataset', choices=['imagenet-val', 'imagenet-trainaug', 'holidays', 'copydays', 'ukbench'],
                        default='imagenet-val', help='which evaluation to make')
    # 是否打乱评估数据集
    parser.add_argument('--shuffle', action='store_true', help='shuffle dataset before evaluation')
    # 删除位置
    parser.add_argument('--expdir', default='experiments/resnet50/finetune500_whitened/holidays500',
                        help='evaluation destination directory')
    # 对于评估应该用不到
    parser.add_argument('--resume-epoch', default=-1, type=int, help='resume epoch (-1: last, 0: from scratch)')
    # 预训练模型位置
    parser.add_argument('--resume-from', default=None, help='resume checkpoint file/folder')
    # 输入图像大小
    parser.add_argument('--input-size', default=500, type=int, help='images input size')
    # 是否进行裁剪
    parser.add_argument('--input-crop', default='rect', choices=['square', 'rect'], help='crop the input or not')
    # 批量处理大小
    parser.add_argument('--batch-size', default=8, type=int, help='batch size')
    # 基准网络
    parser.add_argument('--backbone', default='resnet50', choices=backbone_list, help='backbone architecture')
    # 基准网络预训练模型, 对于评估应该用不到
    parser.add_argument('--pretrained-backbone', action='store_const', const='imagenet', help='use pretrained backbone')
    # GeM池化指数
    parser.add_argument('--pooling-exponent', default=None, type=float,
                        help='pooling exponent in GeM pooling (default: use value from checkpoint)')
    # CPU还是GPU操作
    parser.add_argument('--no-cuda', action='store_true', help='do not use CUDA')
    # ImageNet数据根路径
    parser.add_argument('--imagenet-path', default='data/ilsvrc2012', help='ImageNet data root')
    # INRIA Holidays数据根路径
    parser.add_argument('--holidays-path', default='data/Holidays', help='INRIA Holidays data root')
    # UKBench数据根路径
    parser.add_argument('--UKBench-path', default='data/UKBench', help='UKBench data root')
    # INRIA Copydays数据根路径
    parser.add_argument('--copydays-path', default='data/Copydays', help='INRIA Copydays data root')
    # 干扰图片列表文件路径
    parser.add_argument('--distractors-list', default='data/distractors.txt', help='list of distractor images')
    # 干扰图片路径
    parser.add_argument('--distractors-path', default='data/distractors', help='path to distractor images')
    # 干扰图片采样数目
    parser.add_argument('--num_distractors', default=0, type=int, help='number of distractor images.')
    # ???
    parser.add_argument('--preload-dir-imagenet', default=None,
                        help='preload imagenet in this directory (useful for slow networks')
    # 数据加载器线程数
    parser.add_argument('--workers', default=20, type=int, help='number of data-fetching workers')
    # 是否不保存结果, 默认为保存
    parser.add_argument('--dry', action='store_true', help='do not store anything')

    args = parser.parse_args()

    run(args)
