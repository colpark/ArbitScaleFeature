from torch.utils.data import Dataset
from torchvision import transforms
import datasets
from datasets import register
import torchvision.transforms.functional as TF
from PIL import Image


'''@register('sr_dataset')
class SRDataset(Dataset):

    def __init__(self, imageset, resize, downsample_ratio_2, downsample_ratio_1 = 1):
        self.imageset = datasets.make(imageset)
        self.resize = resize
        self.downsample_ratio_1 = downsample_ratio_1
        self.downsample_ratio_2 = downsample_ratio_2
        
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.imageset)

    def __getitem__(self, idx):
        pil_img = self.imageset[idx]
        x = self.transform(pil_img) 

        if downsample_ratio_1 != 1:
            downsample_size_1 = (
                int(x.shape[1]//self.downsample_ratio_1),
                int(x.shape[2]//self.downsample_ratio_1),
            )
            x = transforms.functional.resize(pil_img, downsample_size)
            x = transforms.ToTensor()(x)
        
        # y: downsampled version of x (low-res)
        downsample_size_1 = (
            int(x.shape[1]//self.downsample_ratio),
            int(x.shape[2]//self.downsample_ratio),
        )
        y = transforms.functional.resize(pil_img, downsample_size)  # PIL image
        y = transforms.ToTensor()(y)  # convert to tensor
        gt = {'hr': x, 'lr': y}
    
        return {'inp': y, 'gt': gt}

from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
'''

@register('sr_dataset')
class SRDataset(Dataset):
    def __init__(self, imageset, resize_1, resize_2):
        """
        Args:
            imageset: dataset spec
            resize_1: tuple (H, W) size for high-res image
            resize_2: tuple (H, W) size for low-res image
        """
        self.imageset = datasets.make(imageset)
        self.resize_1 = resize_1
        self.resize_2 = resize_2

    def __len__(self):
        return len(self.imageset)

    def _resize_image(self, pil_img, target_size):
        img = TF.resize(pil_img, target_size)
        img = TF.center_crop(img, target_size)
        return img

    def __getitem__(self, idx):
        pil_img = self.imageset[idx]

        # High-resolution image
        img_hr_pil = self._resize_image(pil_img, self.resize_1)
        x = TF.to_tensor(img_hr_pil)

        # Low-resolution image
        img_lr_pil = self._resize_image(img_hr_pil, self.resize_2)
        y = TF.to_tensor(img_lr_pil)

        gt = {'hr': x, 'lr': y}
        return {'inp': y, 'gt': gt}

