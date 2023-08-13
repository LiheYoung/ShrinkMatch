from torchvision import datasets, transforms
from PIL import Image
import random
from dataset.randaugment import RandAugmentMC


class CIFAR10Pair(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, multicrop_transform=None, repeat=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.repeat = repeat
        self.multicrop_transform = multicrop_transform

        class_dict = {}
        for index, label in enumerate(self.targets):
            if label in class_dict:
                class_dict[label].append(index)
            else:
                class_dict[label] = [index]
        self.class_dict = class_dict

    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]
        img1 = Image.fromarray(img1)
        if self.repeat:
            img2 = img1
        else:
            img2 = Image.fromarray(self.data[random.choice(self.class_dict[target])])
        
        if isinstance(self.transform, list):
            self.transform1 = self.transform[0]
            self.transform2 = self.transform[1]
        else:
            self.transform1 = self.transform
            self.transform2 = self.transform

        img1_list = [self.transform1(img1)]
        img2_list = [self.transform2(img2)]
    
        if not self.multicrop_transform is None:
            for _ in range(6):
                img1_list.append(self.multicrop_transform(img1))
                # img2_list.append(self.multicrop_transform(img2))

        return img1_list, img2_list, target


class CIFAR100Pair(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, multicrop_transform=None, repeat=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.repeat = repeat
        self.multicrop_transform = multicrop_transform

        class_dict = {}
        for index, label in enumerate(self.targets):
            if label in class_dict:
                class_dict[label].append(index)
            else:
                class_dict[label] = [index]
        self.class_dict = class_dict

    def __getitem__(self, index):
        img1, target = self.data[index], self.targets[index]
        img1 = Image.fromarray(img1)
        if self.repeat:
            img2 = img1
        else:
            img2 = Image.fromarray(self.data[random.choice(self.class_dict[target])])

        if isinstance(self.transform, list):
            self.transform1 = self.transform[0]
            self.transform2 = self.transform[1]
        else:
            self.transform1 = self.transform
            self.transform2 = self.transform

        img1_list = [self.transform1(img1)]
        img2_list = [self.transform2(img2)]
    
        if not self.multicrop_transform is None:
            for _ in range(6):
                img1_list.append(self.multicrop_transform(img1))
                # img2_list.append(self.multicrop_transform(img2))

        return img1_list, img2_list, target


def get_train_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform


def get_rand_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size = 32
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
        size = 96
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        size = 32

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=size, padding=int(size*0.125), padding_mode='reflect'),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform


def get_test_augment(dataset):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'stl10':
        mean = (0.4408, 0.4279, 0.3867)
        std = (0.2682, 0.2610, 0.2686)
    else:
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return test_transform
