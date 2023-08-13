import argparse
import logging
import time

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from dataset.data import *
from dataset.semi_data import *
from network.google_wide_resnet import wide_resnet28w2, wide_resnet28w8
from network.shrinkmatch import ShrinkMatch
from util.accuracy import accuracy
from util.dist_init import *
from util.log import init_log
from util.meter import *
from util.torch_dist_sum import *


parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, default=23456)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--epochs', type=int, default=1024)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--label_per_class', type=int, default=10)
parser.add_argument('--threshold', type=float, default=0.95)
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--DA', default=False, action='store_true')
parser.add_argument('--output_path', type=str, default='./exp')
parser.add_argument('--local_rank', default=0, type=int)

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

epochs = args.epochs


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


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


def train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, local_rank, certain_ratio_ema, rank, logger):
    batch_time = AverageMeter('Time', ':.3f')
    data_time = AverageMeter('Data', ':.3f')
    losses_x = AverageMeter('Loss x', ':.3f')
    losses_u = AverageMeter('Loss u', ':.3f')
    losses_shrink = AverageMeter('Loss shrink', ':.3f')
    losses_shrink_weighted = AverageMeter('Loss shrink re-weighted', ':.3f')
    removed_class_idx = AverageMeter('Removed class index', ':.2f')
    mask_ratio = AverageMeter('Mask ratio', ':.3f')
    
    progress = ProgressMeter(
        n_iters_per_epoch,
        [batch_time, data_time, losses_x, losses_u, losses_shrink, losses_shrink_weighted, removed_class_idx, mask_ratio],
        prefix="Epoch {}, Iter: ".format(epoch))
    
    end = time.time()

    dltrain_x.sampler.set_epoch(epoch)
    dltrain_u.sampler.set_epoch(epoch)
    dl_x, dl_u = iter(dltrain_x), iter(dltrain_u)

    model.train()
    certain_ratio_ema = None
    for i in range(n_iters_per_epoch):

        data_time.update(time.time() - end)

        ims_x_weak, lbs_x, index_x = next(dl_x)
        (ims_u_weak, ims_u_strong), lbs_u_real = next(dl_u)

        lbs_x = lbs_x.cuda(local_rank, non_blocking=True)
        index_x = index_x.cuda(local_rank, non_blocking=True)
        lbs_u_real = lbs_u_real.cuda(local_rank, non_blocking=True)
        ims_x_weak = ims_x_weak.cuda(local_rank, non_blocking=True)
        ims_u_weak = ims_u_weak.cuda(local_rank, non_blocking=True)
        ims_u_strong = ims_u_strong.cuda(local_rank, non_blocking=True)

        logits_x, pseudo_label, logits_u_s, logits_u_s_aux = model(ims_x_weak, ims_u_weak, ims_u_strong, args=args)
        loss_x = F.cross_entropy(logits_x, lbs_x, reduction='mean')
        
        pseudo_label = pseudo_label.detach()
        max_probs, _ = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(args.threshold).float()
        loss_u = (torch.sum(-F.log_softmax(logits_u_s, dim=1) * pseudo_label.detach(), dim=1) * mask).mean()
        
        loss_u_shrink, removed_class_idx_cur = shrink_loss(pseudo_label, logits_u_s_aux, args.threshold)
        
        # for our loss reweighting principle 2
        if certain_ratio_ema is None:
            certain_ratio_ema = mask.mean().item()
        else:
            certain_ratio_ema = certain_ratio_ema * 0.999 + mask.mean().item() * 0.001
        
        B = logits_u_s.shape[0]
        
        # loss_u_shrink is multiplied by certain_ratio_ema for our loss reweighting principle 2
        loss = loss_x + loss_u + (loss_u_shrink / B) * certain_ratio_ema
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - end)
        end = time.time()

        losses_x.update(loss_x.item())
        losses_u.update(loss_u.item())
        losses_shrink.update(loss_u_shrink / B)
        losses_shrink_weighted.update(loss_u_shrink / B * certain_ratio_ema)
        removed_class_idx.update(sum(removed_class_idx_cur) / (len(removed_class_idx_cur) + 1e-5))
        mask_ratio.update(mask.mean().item())
        
        if rank == 0 and i % 100 == 0:
            logger.info(progress.format(i))
    
    return certain_ratio_ema


@torch.no_grad()
def test(model,  test_loader, local_rank):
    model.eval()
    # ---------------------- Test --------------------------
    top1 = AverageMeter('top1', ':.3f')
    top5 = AverageMeter('top5', ':.3f')
    ema_top1 = AverageMeter('ema_top1', ':.3f')
    ema_top5 = AverageMeter('ema_top5', ':.3f')
    with torch.no_grad():
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)
            
            out = model.module.encoder_q(image)
            acc1, acc5 = accuracy(out, label, topk=(1, 5))
            top1.update(acc1[0], image.size(0))
            top5.update(acc5[0], image.size(0))

            ema_out = model.module.ema(image)
            ema_acc1, ema_acc5 = accuracy(ema_out, label, topk=(1, 5))
            ema_top1.update(ema_acc1[0], image.size(0))
            ema_top5.update(ema_acc5[0], image.size(0))

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, top1.sum, top1.count, top5.sum, top5.count)
    top1_acc = sum(sum1.float()) / sum(cnt1.float())
    top5_acc = sum(sum5.float()) / sum(cnt5.float())

    sum1, cnt1, sum5, cnt5 = torch_dist_sum(local_rank, ema_top1.sum, ema_top1.count, ema_top5.sum, ema_top5.count)
    ema_top1_acc = sum(sum1.float()) / sum(cnt1.float())
    ema_top5_acc = sum(sum5.float()) / sum(cnt5.float())

    return top1_acc, top5_acc, ema_top1_acc, ema_top5_acc


def main():
    rank, world_size = setup_distributed(port=args.port)
    local_rank = rank
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    if local_rank == 0:
        logger.info(args)

    batch_size = 64 // world_size
    n_iters_per_epoch = 1024
    lr = 0.03
    mu = 7

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    dltrain_x, dltrain_u = get_fixmatch_data(dataset=args.dataset, label_per_class=args.label_per_class, 
                                             batch_size=batch_size, n_iters_per_epoch=n_iters_per_epoch, 
                                             mu=mu, dist=True, return_index=True)

    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=get_test_augment('cifar10'))
        num_classes = 10
    else:
        assert args.dataset == 'cifar100'
        test_dataset = datasets.CIFAR100(root='data', train=False, download=True, transform=get_test_augment('cifar100'))
        num_classes = 100

    if args.dataset == 'cifar100':
        weight_decay = 1e-3
        base_model = wide_resnet28w8()
    else:
        assert args.dataset == 'cifar10'
        weight_decay = 5e-4
        base_model = wide_resnet28w2()
    
    model = ShrinkMatch(base_encoder=base_model, num_classes=num_classes, args=args)
    model.cuda()
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.SGD(grouped_parameters, lr=lr, momentum=0.9, nesterov=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, epochs*n_iters_per_epoch)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True, sampler=test_sampler)

    best_acc1 = best_acc5 = 0
    best_ema1 = best_ema5 = 0
    certain_ratio_ema = None

    if rank==0:
        os.makedirs(args.output_path, exist_ok=True)
    
    if os.path.exists(args.checkpoint):
        checkpoint =  torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        if 'best_acc1' in checkpoint.keys():
            best_acc1 = checkpoint['best_acc1']
            best_acc5 = checkpoint['best_acc5']
            best_ema1 = checkpoint['best_ema1']
            best_ema5 = checkpoint['best_ema5']
        if 'certain_ratio_ema' in checkpoint.keys():
            certain_ratio_ema = checkpoint['certain_ratio_ema']
    else:
        start_epoch = 0
    
    for epoch in range(start_epoch, epochs):
        certain_ratio_ema = train(model, optimizer, scheduler, dltrain_x, dltrain_u, epoch, n_iters_per_epoch, local_rank, certain_ratio_ema, rank, logger)
        top1_acc, top5_acc, ema_top1_acc, ema_top5_acc = test(model, test_loader, local_rank)

        best_acc1 = max(top1_acc.item(), best_acc1)
        best_acc5 = max(top5_acc.item(), best_acc5)
        best_ema1 = max(ema_top1_acc.item(), best_ema1)
        best_ema5 = max(ema_top5_acc.item(), best_ema5)

        if rank == 0:
            logger.info('<====== Evaluation ======> Epoch: {}, Acc@1 {top1_acc:.3f} Acc@5 {top5_acc:.3f} Best_Acc@1 {best_acc:.3f} '
                        'Best_Acc@5 {best_acc5:.3f}'.format(epoch, top1_acc=top1_acc, top5_acc=top5_acc, best_acc=best_acc1, best_acc5=best_acc5))
            logger.info('<====== Evaluation ======> Epoch: {}, EMA@1 {top1_acc:.3f} EMA@5 {top5_acc:.3f} Best_EMA@1 {best_acc:.3f} '
                        'Best_EMA@5 {best_acc5:.3f}'.format(epoch, top1_acc=ema_top1_acc, top5_acc=ema_top5_acc, best_acc=best_ema1, best_acc5=best_ema5))
            print()
            
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1,
                    'best_acc1': best_acc1,
                    'best_acc5': best_acc5,
                    'best_ema1': best_ema1,
                    'best_ema5': best_ema5,
                    'certain_ratio_ema': certain_ratio_ema
                }, os.path.join(args.output_path, 'latest.pth'))


if __name__ == "__main__":
    main()
