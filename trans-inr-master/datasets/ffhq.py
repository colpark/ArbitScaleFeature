import os
from PIL import Image

from torch.utils.data import Dataset

from datasets import register


@register('ffhq')
class FFHQ(Dataset):

    def __init__(self, root_path, split):
        if split == 'train':
            s, t = 1, 41600
        elif split == 'val':
            s, t = 41601, 46800
        elif split == 'test':
            s, t = 46801, 52000
        self.data = []
        for i in range(s, t + 1):
            path = os.path.join(root_path, f'{i:06}.png')
            self.data.append(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Image.open(self.data[idx])
