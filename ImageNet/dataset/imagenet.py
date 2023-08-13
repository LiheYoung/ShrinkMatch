import os
import torchvision
import torchvision.transforms as transforms
from PIL import Image


class ImagenetPercentV2(torchvision.datasets.ImageFolder):
    def __init__(self, root, percent, mode, aug=None, return_index=False):
        super().__init__(root, aug)

        self.return_index = return_index
        if percent == 0.01:
            if mode == 'labeled':
                semi_file = 'semi_files/split_1p_index.txt'
            elif mode == 'unlabeled':
                semi_file = 'semi_files/split_99p_index.txt'
        elif percent == 0.1:
            if mode == 'labeled':
                semi_file = 'semi_files/split_10p_index.txt'
            elif mode == 'unlabeled':
                semi_file = 'semi_files/split_90p_index.txt'
        else:
            assert mode == 'val'
        
        self.root = os.path.join(root, 'val' if mode == 'val' else 'train')
        classes, class_to_idx = self._find_classes(self.root)
        
        self.samples = []
        
        if mode == 'val':
            classes.sort()
            for cls_name in classes:
                filenames = os.listdir(os.path.join(self.root, cls_name))
                filenames.sort()
                for filename in filenames:
                    self.samples.append((class_to_idx[cls_name], os.path.join(self.root, cls_name, filename)))
            assert len(self.samples) == 50000
        
        else:
            filenames = []
            with open(semi_file) as f:
                for line in f:
                    filenames.append(line.strip())
            
            for filename in filenames:
                cls_name = filename.split('_')[0]
                self.samples.append((class_to_idx[cls_name], os.path.join(self.root, cls_name, filename)))
        
        if aug is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = aug

    def __len__(self):
        return self.samples.__len__()

    def __getitem__(self, index):
        label, name = self.samples[index]
        img = Image.open(name).convert('RGB')

        if isinstance(self.transform, list):
            transformed_image = [t(img) for t in self.transform]
        else:
            transformed_image = self.transform(img)
        
        if self.return_index:
            return transformed_image, label, index
        return transformed_image, label

    def _find_classes(self, directory):
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx