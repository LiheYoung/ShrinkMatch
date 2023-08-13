from torchvision import datasets
from dataset.data import *
from dataset.sampler import *
import numpy as np


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_index=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.return_index:
            return img, target, index
        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, return_index=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        
        if self.return_index:
            return img, target, index
        return img, target


class STL10SSL(datasets.STL10):
    def __init__(self, root, indexs=None, split='train',
                 transform=None, target_transform=None,
                 download=False, return_index=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
        self.return_index = return_index

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)
        
        if self.return_index:
            return img, int(target), index
        return img, int(target)


def x_u_split(label_per_class, num_classes, labels, include_labeled=True):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []

    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labeled = idx[:label_per_class]
        labeled_idx.append(labeled)
        if include_labeled:
            unlabeled_idx.append(idx)
        else:
            unlabeled = idx[label_per_class:]
            unlabeled_idx.append(unlabeled)
    
    labeled_idx = np.concatenate(labeled_idx)
    unlabeled_idx = np.concatenate(unlabeled_idx)
    return labeled_idx, unlabeled_idx


class TwoCropsTransform:
    """Take 2 random augmentations of one image."""
    
    def __init__(self,trans_weak,trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong
    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class MultiCropsTransform:
    """Take 2 random augmentations of one image."""
    def __init__(self,trans):
        self.trans = trans
    def __call__(self, x):
        imgs = [ t(x) for t in self.trans ]
        return imgs


def get_fixmatch_data(dataset='cifar10', label_per_class=10, batch_size=64, n_iters_per_epoch=1024, mu=7, dist=False, return_index=False):
    weak_augment = get_train_augment(dataset)
    rand_augment = get_rand_augment(dataset)

    pair_transform = TwoCropsTransform(weak_augment, rand_augment)

    if dataset == 'cifar10':
        data = datasets.CIFAR10(root='data', download=True, transform=weak_augment)
        num_classes = 10
    elif dataset == 'cifar100':
        data = datasets.CIFAR100(root='data', download=True, transform=weak_augment)
        num_classes = 100
    elif dataset == 'stl10':
        data = datasets.STL10(root='data', split='train', download=True, transform=weak_augment)
        data.targets = data.labels
        num_classes = 10

    labeled_idx, unlabeled_idx = x_u_split(label_per_class=label_per_class, num_classes=num_classes, labels=data.targets)

    if dataset == 'cifar10':
        loader = CIFAR10SSL
    else:
        assert dataset == 'cifar100'
        loader = CIFAR100SSL
    
    ds_x = loader(
        root='data',
        indexs=labeled_idx,
        transform=weak_augment,
        return_index=return_index,
    )

    ds_u = loader(
        root='data',
        indexs=unlabeled_idx,
        transform=pair_transform
    )

    if not dist:
        sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    else:
        word_size = torch.distributed.get_world_size()
        sampler_x = DistributedReplacementSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size * word_size)

    dl_x = torch.utils.data.DataLoader(
        ds_x,
        sampler=sampler_x,
        batch_size=batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    if not dist:
        sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    else:
        word_size = torch.distributed.get_world_size()
        sampler_u = DistributedReplacementSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size * word_size)

    dl_u = torch.utils.data.DataLoader(
        ds_u,
        sampler=sampler_u,
        batch_size=mu*batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    if dist:
        return dl_x, dl_u
    
    return dl_x, dl_u
