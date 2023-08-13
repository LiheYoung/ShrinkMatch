import shutil
import torch


def torch_dist_sum(gpu, *args):
    process_group = torch.distributed.group.WORLD
    tensor_args = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            tensor_arg = arg.clone().reshape(-1).detach().cuda(gpu)
        else:
            tensor_arg = torch.tensor(arg).reshape(-1).cuda(gpu)
        torch.distributed.all_reduce(tensor_arg, group=process_group)
        tensor_args.append(tensor_arg)
    
    return tensor_args


def save_checkpoint(state, is_best, epoch, args, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')
    if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
        shutil.copyfile(filename, 'checkpoint_{:04d}.pth.tar'.format(epoch))
    if not args.cos:
        if (epoch + 1) in args.schedule:
            shutil.copyfile(filename, 'checkpoint_{:04d}.pth.tar'.format(epoch))


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name}: {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def format(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return ', '.join(entries)

    def _get_batch_fmtstr(self, num_batches):
        return '{:}/' + str(num_batches)


class InstantMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.val = 0

    def update(self, val):
        self.val = val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
