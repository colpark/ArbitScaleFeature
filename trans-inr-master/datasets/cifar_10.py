import os
from PIL import Image

from torch.utils.data import Dataset

from datasets import register


@register('cifar_10')
class CIFAR10(Dataset):

    def __init__(self, root_path, split):
        if split == 'train':
            s, t = 1, 49999
        elif split == 'val':
            s, t = 50000, 54999
        elif split == 'test':
            s, t = 55000, 59999
        self.data = []
        for i in range(s, t + 1):
            path = os.path.join(root_path, 'cifar-10', f'{i:06}.jpg')
            self.data.append(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Image.open(self.data[idx])
