import argparse
import logging
import math
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset.imagenet import *

import backbone as backbone_models
from models.shrinkmatch import ShrinkMatch
from utils import utils, lr_schedule, get_norm, dist_utils
from utils.log import init_log
import dataset.transforms as data_transforms
from engine import validate


backbone_model_names = sorted(name for name in backbone_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(backbone_models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--trainindex_x', default=None, type=str, metavar='PATH',
                    help='path to train annotation_x (default: None)')
parser.add_argument('--trainindex_u', default=None, type=str, metavar='PATH',
                    help='path to train annotation_u (default: None)')
parser.add_argument('--arch', metavar='ARCH', default='ShrinkMatch',
                    help='model architecture')
parser.add_argument('--backbone', default='resnet50_encoder',
                    choices=backbone_model_names,
                    help='model architecture: ' +
                        ' | '.join(backbone_model_names) +
                        ' (default: resnet50_encoder)')
parser.add_argument('--cls', default=1000, type=int, metavar='N',
                    help='number of classes')
parser.add_argument('--port', default=23456, type=int, help='dist init port')                    
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--warmup-epoch', default=0, type=int, metavar='N',
                    help='number of epochs for learning warmup')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='use nesterov momentum')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 100)')
parser.add_argument('--eval-freq', default=1, type=int,
                    metavar='N', help='evaluation epoch frequency (default: 1)')
parser.add_argument('--output-path', default='', type=str, metavar='PATH',
                    help='path to output logs and checkpoints (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                    help='path to pretrained model (default: none)')
parser.add_argument('--self-pretrained', default='', type=str, metavar='PATH',
                    help='path to MoCo pretrained model (default: none)')
parser.add_argument('--super-pretrained', default='', type=str, metavar='PATH',
                    help='path to supervised pretrained model (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--data-path', type=str, default=None,
                    help='path of ImageNet data')
parser.add_argument('--anno-percent', type=float, default=0.1,
                    help='number of labeled data')
parser.add_argument('--split-seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--mu', default=5, type=int,
                    help='coefficient of unlabeled batch size')
parser.add_argument('--lambda-u', default=10, type=float,
                    help='coefficient of unlabeled loss')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='pseudo label threshold')
parser.add_argument('--eman', action='store_true', default=False,
                    help='use EMAN')
parser.add_argument('--ema-m', default=0.999, type=float,
                    help='EMA decay rate')
parser.add_argument('--norm', default='None', type=str,
                    help='the normalization for backbone (default: None)')
# online_net.backbone for BYOL
parser.add_argument('--moco-path', default=None, type=str)
parser.add_argument('--model-prefix', default='encoder_q', type=str,
                    help='the model prefix of self-supervised pretrained state_dict')
parser.add_argument('--st', type=float, default=0.1)
parser.add_argument('--tt', type=float, default=0.1)
parser.add_argument('--c_smooth', type=float, default=1.0)
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--lambda_in', type=float, default=1)
parser.add_argument('--randaug', default=False, action='store_true')
parser.add_argument('--stack', default=False, action='store_true')
parser.add_argument('--local_rank', default=0, type=int)
args = parser.parse_args()


def shrink_loss(pseudo_label, logits_u_s, conf_thresh):
    removed_class_idx = []
    loss_u_shrink_batch = 0
    
    B, C = pseudo_label.shape
    
    max_probs = pseudo_label.max(dim=-1)[0]
    mask = pseudo_label.ge(conf_thresh).float()
    
    sorted_prob_w, sorted_idx = pseudo_label.topk(C, dim=-1, sorted=True)
    # organize logit_s same as the sorted_prob_w
    sorted_logits_s = logits_u_s.gather(dim=-1, index=sorted_idx)
    
    if mask.mean().item() == 1: # no uncertain samples to shrink
        loss_u_shrink_batch = 0
    else:
        for b in range(B):
            if max_probs[b] >= conf_thresh: # skip certain samples
                continue
            # iteratively remove classes to enhance confidence until satisfying the confidence threshold
            for c in range(2, C):
                # new confidence in the shrunk class space (classes ranging from 1 ~ (c-1) are removed)
                sub_conf = sorted_prob_w[b, 0] / (sorted_prob_w[b, 0] + sorted_prob_w[b, c:].sum())
                
                # break either when satifying the threshold or traversing to the final class (with smallest value)
                if (sub_conf >= conf_thresh) or (c == C - 1):
                    sub_logits_s = torch.cat([sorted_logits_s[b, :1], sorted_logits_s[b, c:]], dim=0)
                    
                    loss_u_shrink = F.cross_entropy(sub_logits_s[None, ], torch.zeros(1).long().cuda(), reduction='none')[0] 
                    # for our loss reweighting principle 1
                    loss_u_shrink *= max_probs[b] * ((sub_conf >= conf_thresh))
                    
                    loss_u_shrink_batch += loss_u_shrink
                    
                    removed_class_idx.append(c)
                    
                    break
    
    return loss_u_shrink_batch, removed_class_idx


def main_worker():
    best_acc1 = 0
    best_acc5 = 0

    rank, world_size = dist_utils.dist_init(port=args.port)
    args.gpu = rank
    args.world_size = world_size
    args.distributed = True
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    if rank == 0:
        logger.info(args)
    
    train_dataset_x, train_dataset_u, val_dataset = get_imagenet_ssl()

    # Data loading code
    train_sampler = DistributedSampler

    train_loader_x = DataLoader(
        train_dataset_x,
        sampler=train_sampler(train_dataset_x),
        batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    train_loader_u = DataLoader(
        train_dataset_u,
        sampler=train_sampler(train_dataset_u),
        batch_size=args.batch_size * args.mu,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        sampler=train_sampler(val_dataset),
        batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    if rank == 0:
        logger.info("=> creating model '{}' with backbone '{}'".format(args.arch, args.backbone))
    norm = get_norm(args.norm)
    model = ShrinkMatch(
        backbone_models.__dict__[args.backbone],
        eman=args.eman,
        momentum=args.ema_m,
        norm=norm,
        K=len(train_dataset_x),
        args=args
    )
    
    if args.moco_path is not None:
        checkpoint = torch.load(args.moco_path, map_location="cpu")
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            if k.startswith('module.encoder_q'):
                state_dict[k.replace('module.encoder_q', 'backbone')] = state_dict[k]
            del state_dict[k]
        
        for k in list(state_dict.keys()):
            if 'backbone.fc.0' in k:
                state_dict[k.replace('backbone.fc.0','head.0')] = state_dict[k]
                del state_dict[k]
            if 'backbone.fc.2' in k:
                state_dict[k.replace('backbone.fc.2','head.2')] = state_dict[k]            
                del state_dict[k]
        
        model.main.load_state_dict(state_dict=state_dict, strict=False)

        for param, param_m in zip(model.main.parameters(), model.ema.parameters()):
            param_m.data.copy_(param.data)  
            param_m.requires_grad = False

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    certain_ratio_ema = None
    
    if os.path.exists(args.checkpoint):
        checkpoint =  torch.load(args.checkpoint, map_location='cpu')
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'best_acc1' in checkpoint.keys():
            best_acc1 = checkpoint['best_acc1']
            best_acc5 = checkpoint['best_acc5']
        if 'certain_ratio_ema' in checkpoint.keys():
            certain_ratio_ema = checkpoint['certain_ratio_ema']
        if rank == 0:
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.checkpoint, checkpoint['epoch']))
    else:
        if rank == 0:
            logger.info("=> no checkpoint found at '{}'".format(args.checkpoint))
    
    cudnn.benchmark = True
    
    if args.evaluate:
        acc1, acc5 = validate(val_loader, model, criterion, args)
        if rank == 0:
            logger.info('<====== Evaluation ======> Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc1, acc5))
    else:
        for epoch in range(args.start_epoch, args.epochs):
            if epoch >= args.warmup_epoch:
                lr_schedule.adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            certain_ratio_ema = train(train_loader_x, train_loader_u, model, optimizer, epoch, certain_ratio_ema, args, rank, logger)
            
            if (epoch + 1) % args.eval_freq == 0:
                # evaluate on validation set
                acc1, acc5 = validate(val_loader, model, criterion, args)
                # remember best acc@1 and save checkpoint
                best_acc1 = max(acc1.item(), best_acc1)
                best_acc5 = max(acc5.item(), best_acc5)
                is_best = (best_acc1 == acc1)

            if rank == 0:
                logger.info('<====== Evaluation ======> Epoch:{}, Acc@1 {:.3f} Acc@5 {:.3f} Best_Acc@1 {:.3f} '
                            'Best_Acc@5 {:.3f}'.format(epoch, acc1, acc5, best_acc1, best_acc5))
                print()
                torch.save({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'certain_ratio_ema': certain_ratio_ema
                }, os.path.join(args.output_path, 'latest.pth'))
                if is_best:
                    shutil.copy(os.path.join(args.output_path, 'latest.pth'), os.path.join(args.output_path, 'best.pth'))


def train(train_loader_x, train_loader_u, model, optimizer, epoch, certain_ratio_ema, args, rank, logger):
    batch_time = utils.AverageMeter('Time', ':.3f')
    data_time = utils.AverageMeter('Data', ':.3f')
    losses = utils.AverageMeter('Loss', ':.3f')
    losses_x = utils.AverageMeter('Loss x', ':.3f')
    losses_u = utils.AverageMeter('Loss u', ':.3f')
    losses_in = utils.AverageMeter('Loss in', ':.3f')
    losses_shrink = utils.AverageMeter('Loss shrink', ':.3f')
    losses_shrink_weighted = utils.AverageMeter('Loss shrink re-weighted', ':.3f')
    top1_x = utils.AverageMeter('Acc_x@1', ':.3f')
    top5_x = utils.AverageMeter('Acc_x@5', ':.3f')
    top1_u = utils.AverageMeter('Acc_u@1', ':.3f')
    top5_u = utils.AverageMeter('Acc_u@5', ':.3f')
    top1_u_masked = utils.AverageMeter('Acc_u_masked@1', ':.3f')
    mask_ratio = utils.AverageMeter('Mask ratio', ':.3f')
    removed_class_idx = utils.AverageMeter('Removed class idx', ':.2f')
    
    curr_lr = utils.InstantMeter('LR', ':.3f')
    progress = utils.ProgressMeter(
        len(train_loader_u),
        [curr_lr, batch_time, data_time, losses, losses_x, losses_u, losses_in, losses_shrink, losses_shrink_weighted, 
         top1_x, top5_x, top1_u, top5_u, top1_u_masked, removed_class_idx, mask_ratio],
        prefix="Epoch {}, Iter: ".format(epoch))

    epoch_x = epoch * math.ceil(len(train_loader_u) / len(train_loader_x))
    if args.distributed:
        train_loader_x.sampler.set_epoch(epoch_x)
        train_loader_u.sampler.set_epoch(epoch)

    train_iter_x = iter(train_loader_x)
    # switch to train mode
    model.train()
    if args.eman:
        if rank == 0:
            logger.info("setting the ema model to eval mode")
        if hasattr(model, 'module'):
            model.module.ema.eval()
        else:
            model.ema.eval()

    end = time.time()
    
    for i, (images_u, targets_u) in enumerate(train_loader_u):
        try:
            images_x, targets_x, index = next(train_iter_x)
        except Exception:
            epoch_x += 1
            if args.distributed:
                train_loader_x.sampler.set_epoch(epoch_x)
            train_iter_x = iter(train_loader_x)
            images_x, targets_x, index = next(train_iter_x)
        
        images_u_w, images_u_s = images_u
        # measure data loading time
        data_time.update(time.time() - end)

        images_x = images_x.cuda(args.gpu, non_blocking=True)
        images_u_w = images_u_w.cuda(args.gpu, non_blocking=True)
        images_u_s = images_u_s.cuda(args.gpu, non_blocking=True)
        targets_x = targets_x.cuda(args.gpu, non_blocking=True)
        targets_u = targets_u.cuda(args.gpu, non_blocking=True)
        index = index.cuda(args.gpu, non_blocking=True)

        # warmup learning rate
        if epoch < args.warmup_epoch:
            warmup_step = args.warmup_epoch * len(train_loader_u)
            curr_step = epoch * len(train_loader_u) + i + 1
            lr_schedule.warmup_learning_rate(optimizer, curr_step, warmup_step, args)
        curr_lr.update(optimizer.param_groups[0]['lr'])

        # model forward
        logits_x, pseudo_label, logits_u_s, loss_in, logits_u_s_aux = model(images_x, images_u_w, images_u_s, labels=targets_x, 
                                                                            index=index, start_unlabel=epoch>0, args=args)
        max_probs = torch.max(pseudo_label, dim=-1)[0]
        mask = max_probs.ge(args.threshold).float()

        loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')
        loss_u = (torch.sum(-F.log_softmax(logits_u_s,dim=1) * pseudo_label.detach(), dim=1) * mask).mean()
        
        loss_in = loss_in.mean()
        
        loss_u_shrink, removed_class_idx_cur = shrink_loss(pseudo_label, logits_u_s_aux, args.threshold)
        
        # for our loss reweighting principle 2
        if certain_ratio_ema is None:
            certain_ratio_ema = mask.mean().item()
        else:
            certain_ratio_ema = certain_ratio_ema * 0.999 + mask.mean().item() * 0.001
        
        B = logits_u_s.shape[0]
        
        # loss_u_shrink is multiplied by certain_ratio_ema for our loss reweighting principle 2
        loss = loss_x + args.lambda_u * loss_u + args.lambda_in * loss_in + args.lambda_u * (loss_u_shrink / B) * certain_ratio_ema
        
        # measure accuracy and record loss
        losses.update(loss.item())
        losses_x.update(loss_x.item(), images_x.size(0))
        losses_u.update(loss_u.item(), images_u_w.size(0))
        losses_in.update(loss_in.item(), images_u_w.size(0))
        acc1_x, acc5_x = utils.accuracy(logits_x, targets_x, topk=(1, 5))
        top1_x.update(acc1_x[0], logits_x.size(0))
        top5_x.update(acc5_x[0], logits_x.size(0))
        acc1_u, acc5_u = utils.accuracy(pseudo_label, targets_u, topk=(1, 5))
        top1_u.update(acc1_u[0], pseudo_label.size(0))
        top5_u.update(acc5_u[0], pseudo_label.size(0))
        mask_ratio.update(mask.mean().item(), mask.size(0))

        bool_mask = mask.bool()
        acc1_u_masked = sum(pseudo_label.max(1)[1][bool_mask] == targets_u[bool_mask]) / (bool_mask.sum() + 1e-8)
        top1_u_masked.update(acc1_u_masked * 100)
        
        removed_class_idx.update(sum(removed_class_idx_cur) / (len(removed_class_idx_cur) + 1e-5))
        losses_shrink.update(loss_u_shrink / B)
        losses_shrink_weighted.update(loss_u_shrink / B * certain_ratio_ema)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update the ema model
        if hasattr(model, 'module'):
            model.module.momentum_update_ema()
        else:
            model.momentum_update_ema()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if rank == 0 and i % args.print_freq == 0:
            logger.info(progress.format(i))
        
    return certain_ratio_ema


def get_imagenet_ssl(val_type='DefaultVal'):
    transform_x = data_transforms.weak_aug
    weak_transform = data_transforms.weak_aug
    if args.stack:
        strong_transform = data_transforms.stack_aug
    if args.randaug:
        strong_transform = data_transforms.rand_aug
    else:
        strong_transform = data_transforms.moco_aug
    transform_u = data_transforms.TwoCropsTransform(weak_transform, strong_transform)
    transform_val = data_transforms.get_transforms(val_type)

    train_dataset_x = ImagenetPercentV2(root=args.data_path, percent=args.anno_percent, mode='labeled', aug=transform_x, return_index=True)
    train_dataset_u = ImagenetPercentV2(root=args.data_path, percent=args.anno_percent, mode='unlabeled', aug=transform_u)
    val_dataset = ImagenetPercentV2(root=args.data_path, percent=None, mode='val', aug=transform_val)

    return train_dataset_x, train_dataset_u, val_dataset


if __name__ == '__main__':
    main_worker()
