import numpy as np
import torch
import torchvision
import einops
import wandb

from .base_trainer_grad import BaseTrainer
from trainers import register
from utils import make_coord_grid


@register('imgrec_trainer_grad')
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
        warmup_epochs = 5
        min_lr = 9.e-6

        if self.epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (self.epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            t = (self.epoch - warmup_epochs) / (self.cfg['max_epoch'] - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr-min_lr) * (1 + np.cos(np.pi * t))
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        self.log_temp_scalar('lr', lr)

    def _iter_step(self, data, is_train):
        data = {k: v.cuda() for k, v in data.items()}
        gt = data.pop('gt')
        B = gt.shape[0]
    
        hyponet = self.model_ddp(data)
    
        coord = make_coord_grid(gt.shape[-2:], (-1, 1), device=gt.device)
        coord = einops.repeat(coord, 'h w d -> b h w d', b=B)
        pred = hyponet(coord)  # b h w 3
        gt = einops.rearrange(gt, 'b c h w -> b h w c')
        mses = ((pred - gt) ** 2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()
    
        grad_norm = None  # ✅ Define before the if block
    
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
    
            # ✅ Compute gradient norm before step
            grad_norm = self.get_grad_norm(self.model_ddp)
    
            self.optimizer.step()
    
        ret = {'loss': loss.item(), 'psnr': psnr.item()}
        if grad_norm is not None:
            ret['grad_norm'] = grad_norm  # ✅ Now always safe

        return ret


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
            data = {k: v.unsqueeze(0).cuda() for k, v in data.items()}
            gt = data.pop('gt')[0]
            with torch.no_grad():
                hyponet = self.model_ddp(data)
                coord = make_coord_grid(gt.shape[-2:], [-1, 1], device=gt.device)
                coord = coord.unsqueeze(0)
                pred = hyponet(coord)[0]
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
