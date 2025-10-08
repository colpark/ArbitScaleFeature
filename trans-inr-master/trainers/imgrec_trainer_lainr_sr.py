import numpy as np
import torch
import torchvision
import einops
import wandb

from .base_trainer import BaseTrainer
from trainers import register
from utils import make_coord_grid


@register('imgrec_trainer_lainr_sr')
class ImgrecTrainer(BaseTrainer):

    def make_datasets(self):
        super().make_datasets()

        def get_vislist(dataset, n_vis=32):
            ids = torch.arange(n_vis) * (len(dataset) // n_vis)
            return [dataset[i] for i in ids]

        if hasattr(self, 'train_loader'):
            self.vislist_train = get_vislist(self.train_loader.dataset)
        if hasattr(self, 'test_loader'):
            self.vislist_test = get_vislist(self.test_loader.dataset)

    def adjust_learning_rate(self):

        base_lr = self.cfg['optimizer']['args']['lr']
        max_epochs = self.cfg['max_epoch']
        try:
            warmup_epochs = self.cfg['warmup_epochs']
        except Exception as e:
            #print(f"cfg does not specify warmup_epochs, default 5 will be used {e}")
            warmup_epochs = 5         
            
        min_lr = 1.e-8

        if self.epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (self.epoch + 1) / warmup_epochs
        elif self.epoch < max_epochs*0.90:
            lr = base_lr
            # Cosine annealing after warmup
        else:
            t = (self.epoch - max_epochs*0.90) / (max_epochs - max_epochs*0.90)
            lr = min_lr + 0.5 * (base_lr-min_lr) * (1 + np.cos(np.pi * t))
        '''
        base_lr = self.cfg['optimizer']['args']['lr']
        try:
            warmup_epochs = self.cfg['warmup_epochs']
        except Exception as e:
            #print(f"cfg does not specify warmup_epochs, default 5 will be used {e}")
            warmup_epochs = 5
            
            
        min_lr = 1.e-8

        if self.epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (self.epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            t = (self.epoch - warmup_epochs) / (self.cfg['max_epoch'] - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr-min_lr) * (1 + np.cos(np.pi * t))'''
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.log_temp_scalar('lr', lr)

    def add_gaussian_noise_to_grid(self, coord_grid, std=0.01):
       
        noise = torch.randn_like(coord_grid) * std
        noisy_coords = coord_grid + noise
        return noisy_coords

    def sampled_coord_grid(self, coord_grid, gt, keep_ratio = 0.1):
        H, W, D = coord_grid.shape
        B, H_gt, W_gt, C = gt.shape
    
        assert H == H_gt and W == W_gt, "Shape mismatch between coord grid and gt"
    
        device = coord_grid.device
        axis = 'row' if torch.rand(1).item() < 0.5 else 'col'
    
        if axis == 'row':
            num_keep = max(1, int(H * keep_ratio))
            keep_rows = torch.randperm(H, device=device)[:num_keep].sort()[0]
    
            coord_sub = coord_grid[keep_rows, :, :]           # (H', W, D)
            gt_sub = gt[:, keep_rows, :, :]                   # (B, H', W, C)
        else:
            num_keep = max(1, int(W * keep_ratio))
            keep_cols = torch.randperm(W, device=device)[:num_keep].sort()[0]
    
            coord_sub = coord_grid[:, keep_cols, :]           # (H, W', D)
            gt_sub = gt[:, :, keep_cols, :]                   # (B, H, W', C)

        return coord_sub, gt_sub

    import torch

    def perturb_coords(self, grid, p=0.05, std=0.01):
        """
        Efficiently perturb p% of x and y coordinates in grid using a random mask.
    
        Args:
            grid: Tensor of shape (B, H, W, 2), values in [0, 1]
            p: Percentage of coords to perturb (independently for x and y)
            std: Std of Gaussian noise
    
        Returns:
            Perturbed grid of same shape
        """
        H, W, _ = grid.shape
        device = grid.device
    
        grid = grid.clone()
    
        # Create independent random masks for x and y channels
        mask = torch.rand(H, W, 2, device=device) < p  # shape (B, H, W, 2)
    
        # Generate Gaussian noise
        noise = torch.randn_like(grid) * std  # shape (B, H, W, 2)
    
        # Apply noise where mask is True
        grid += noise * mask.float()
    
        # Clamp to [0, 1]
        grid.clamp_(0.0, 1.0)
    
        return grid



    def _iter_step(self, data, is_train):
        alpha = self.alpha
        for k, v in data.items():
            if torch.is_tensor(v):
                data[k] = v.cuda()
        
        gt = data.pop('gt')
        gt = {k: v.cuda() for k, v in gt.items()}
        gt_hr = gt.pop('hr')
        gt_lr = gt.pop('lr')

        gt_hr = einops.rearrange(gt_hr, 'b c h w -> b h w c')
        gt_lr = einops.rearrange(gt_lr, 'b c h w -> b h w c')
        
        B, H, W, C = gt_lr.shape

        hyponet, tokens = self.model_ddp(data)

        if is_train:
            coord_hr = make_coord_grid(gt_hr.shape[-3:-1], (0, 1), device=gt_hr.device)
            coord_hr, gt_hr = self.sampled_coord_grid(coord_hr, gt_hr, 0.2)
            
            coord_lr = make_coord_grid(gt_lr.shape[-3:-1], (0, 1), device=gt_lr.device)
            #print(1.0/(3.0*H))
            coord_lr = self.perturb_coords(coord_lr, self.p, std = 0.9*(1.0/(3.0*H)))
            #coord_lr, gt_lr = self.sampled_coord_grid(coord_lr, gt_lr, 0.9)
            #coord = self.add_gaussian_noise_to_grid(coord)
            coord_hr = einops.repeat(coord_hr, 'h w d -> b h w d', b=B)
            coord_lr = einops.repeat(coord_lr, 'h w d -> b h w d', b=B)
            pred_hr = hyponet(coord_hr, tokens) # b h w 3
            pred_lr = hyponet(coord_lr, tokens) # b h w 3
    
            mses_hr = ((pred_hr - gt_hr)**2).view(B, -1).mean(dim=-1)
            mses_lr = ((pred_lr - gt_lr)**2).view(B, -1).mean(dim=-1)
            mses = alpha*mses_hr + (1.0-alpha)*mses_lr
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses_lr)).mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        else:
            #coord_hr = make_coord_grid(gt_hr.shape[-3:-1], (0, 1), device=gt_hr.device)
            #coord_hr, gt_hr = self.sampled_coord_grid(coord_hr, gt_hr, 0.2)
            
            coord_lr = make_coord_grid(gt_lr.shape[-3:-1], (0, 1), device=gt_lr.device)
            #print(1.0/3.0*H)
            #coord_lr = perturb_coords(coord_lr, 0.3, std = (1.0/3.0*H))
            #coord_lr, gt_lr = self.sampled_coord_grid(coord_lr, gt_lr, 0.9)
            #coord = self.add_gaussian_noise_to_grid(coord)
            #coord_hr = einops.repeat(coord_hr, 'h w d -> b h w d', b=B)
            coord_lr = einops.repeat(coord_lr, 'h w d -> b h w d', b=B)
            #pred_hr = hyponet(coord_hr, tokens) # b h w 3
            pred_lr = hyponet(coord_lr, tokens) # b h w 3
    
            #mses_hr = ((pred_hr - gt_hr)**2).view(B, -1).mean(dim=-1)
            mses_lr = ((pred_lr - gt_lr)**2).view(B, -1).mean(dim=-1)
            #mses = alpha*mses_hr + (1.0-alpha)*mses_lr
            loss = mses_lr.mean()
            psnr = (-10 * torch.log10(mses_lr)).mean()

        return {'loss': loss.item(), 'psnr': psnr.item()}

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    def _gen_vis_result(self, tag, vislist):
        pred_value_range = (0, 1)
        self.model_ddp.eval()
        res = []
        for data in vislist:
            for k, v in data.items():
                if torch.is_tensor(v):
                    data[k] = v.cuda()
            
            gt = data.pop('gt')
            gt = {k: v.cuda() for k, v in gt.items()}
            gt = gt.pop('lr')
            
            with torch.no_grad():
                hyponet, tokens = self.model_ddp(data)
                coord = make_coord_grid(gt.shape[-2:], [0, 1], device=gt.device)
                coord = coord.unsqueeze(0)
                pred = hyponet(coord, tokens)[0]
                pred = einops.rearrange(pred.clamp(*pred_value_range), 'h w c -> c h w')
            res.append(gt)
            res.append(pred)
        res = torch.stack(res)
        res = res.detach().cpu()
        imggrid = torchvision.utils.make_grid(res, normalize=True, value_range=pred_value_range)

        if self.enable_tb:
            self.writer.add_image(tag, imggrid, self.epoch)
        if self.enable_wandb:
            wandb.log({tag: wandb.Image(imggrid)}, step=self.epoch)

    def visualize_epoch(self):
        if hasattr(self, 'vislist_train'):
            self._gen_vis_result('vis_train_dataset', self.vislist_train)
        if hasattr(self, 'vislist_test'):
            self._gen_vis_result('vis_test_dataset', self.vislist_test)
