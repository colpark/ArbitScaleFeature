import os
from PIL import Image

from torch.utils.data import Dataset

from datasets import register


@register('celebahq_sr')
class Celeba(Dataset):

    def __init__(self, root_path, split):
        
        if split == 'train':
            s, t = 1, 27999
        elif split == 'val':
            s, t = 28000, 28999
        elif split == 'test':
            s, t = 29000, 29999
        self.data = []
        for i in range(s, t + 1):         
            
            path = os.path.join(root_path, 'celeba_hq', 'data', f'img_{i:05}.jpg')
            self.data.append(path)
           
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return Image.open(self.data[idx])
