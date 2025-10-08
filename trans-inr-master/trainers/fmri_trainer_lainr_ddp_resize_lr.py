import numpy as np
import torch
import torchvision
import einops
import wandb
import monai.transforms as monai_t

from .base_fmri_trainer_slurm import BaseTrainer
from trainers import register
from utils import make_coord_grid


@register('fmri_trainer_lainr_ddp_resize_lr')
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
            
    def _log(self, *args):
        # append diagnostic lines to file
        with open("testing_log.txt", "a") as f:
            f.write(" ".join(str(a) for a in args) + "\n")

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        max_epochs = self.cfg['max_epoch']
        try:
            warmup_epochs = self.cfg['warmup_epochs']
        except Exception as e:
            #print(f"cfg does not specify warmup_epochs, default 5 will be used {e}")
            warmup_epochs = 10
    
        try:
            min_lr = self.cfg['min_lr_p']*base_lr
        except Exception as e:
            #print(f"cfg does not specify warmup_epochs, default 5 will be used {e}")
            min_lr = 1.e-8

        if self.epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (self.epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            t = (self.epoch - warmup_epochs) / (self.cfg['max_epoch'] - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr-min_lr) * (1 + np.cos(np.pi * t))

        self._log('lr:', lr)
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        '''if self.epoch < warmup_epochs:
            # Linear warmup
            lr = base_lr * (self.epoch + 1) / warmup_epochs
        elif self.epoch < max_epochs*0.95:
            lr = base_lr
            # Cosine annealing after warmup
        else:
            t = (self.epoch - max_epochs*0.95) / (max_epochs - max_epochs*0.95)
            lr = min_lr + 0.5 * (base_lr-min_lr) * (1 + np.cos(np.pi * t))'''
            
        self.log_temp_scalar('lr', lr)

    def add_gaussian_noise_to_grid(self, coord_grid, std=0.01):
       
        noise = torch.randn_like(coord_grid) * std
        noisy_coords = coord_grid + noise
        return noisy_coords

    def transform_sequence(self, seq, resize, is_train): # 250814 jubin change
        B, C, H, W, D, T = seq.shape
        spatial_size = [resize, resize, resize]
        resized_shape = [B, C] + spatial_size + [T]
        resize_transform = monai_t.Resize(spatial_size=(spatial_size))
        transforms = [resize_transform]
        
        if self.cfg['use_augmentation'] and is_train: # 250814 jubin change
            rand_affine = monai_t.RandAffine(prob=0.5, rotate_range=(0.175,)*3, scale_range=(0.1,)*3, # https://docs.monai.io/en/stable/transforms.html#randaffine  # 0.175 rad = 10 degrees
                                             mode="bilinear", padding_mode="zeros", device=seq.device)
            rand_noise = monai_t.RandGaussianNoise(prob=0.5, std=0.05, dtype=seq.dtype) # https://docs.monai.io/en/stable/transforms.html#randgaussiannoise
            scaler = monai_t.ScaleIntensity(minv=0.0, maxv=1.0, dtype=seq.dtype) # min-max scale again after adding Gaussian noise to input.
            transforms += [rand_affine, rand_noise, scaler]
            
        transforms = monai_t.Compose(transforms)
        seq = einops.rearrange(seq, 'b c h w d t -> b t c h w d')
        seq_resized = torch.zeros(resized_shape).to(seq.device)
        for b in range(B):
            for t in range(T):
                aug_seed = torch.randint(0, 10000000, (1,)).item() # 250814 jubin change
                transforms.set_random_state(seed=aug_seed) # set augmentation seed to be the same for all time steps
                seq_resized[b, :, :, :, :, t] = transforms(seq[b, t, :, :, :, :])
        del seq
        return seq_resized
        
    '''def resize_sequence(self, seq, resize):
        B, C, H, W, D, T = seq.shape
        spatial_size = [resize,  resize, resize]
        resized_shape = [B, C] + spatial_size + [T]
        resize_transform = monai_t.Resize(spatial_size=(spatial_size))
        seq = einops.rearrange(seq, 'b c h w d t -> b t c h w d')
        seq_resized = torch.zeros(resized_shape).to(seq.device)
        for b in range(B):
            for t in range(T):
                seq_resized[b, :, :, :, :, t] = resize_transform(seq[b, t, :, :, :, :])
        del seq
        return seq_resized'''
        

    '''# https://docs.monai.io/en/stable/transforms.html#resize
    import monai.transforms as monai_t
    from einops import rearrange
    
    fmri_test = fmri.clone()
    
    B, C, H, W, D, T = fmri.shape
    
    spatial_size = [H//2,  W//2, D//2]
    resized_shape = [B, C] + spatial_size + [T]
    resize_transform = monai_t.Resize(spatial_size=(spatial_size))
    
    fmri_test = fmri_test.cuda()
    
    fmri_test = rearrange(fmri, 'b c h w d t -> b t c h w d')
    fmri_resized = torch.zeros(resized_shape)
    print(fmri_test.shape)
    for b in range(B):
        for t in range(T):
            fmri_resized[b, :, :, :, :, t] = resize_transform(fmri_test[b, t, :, :, :, :])
    print('Resized / original fmri shape:',fmri.shape, fmri_resized.shape)'''

    def _iter_step(self, data, is_train, p=1.0):
        data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
        #data = {k: v.cuda() for k, v in data.items()}
        gt_full = data.pop('fmri_sequence')
        
        '''try:
            resize = self.cfg['resize']
            gt_full = self.resize_sequence(gt_full, resize)
            print(f'gt_full shape is {gt_full.shape}')
        except Exception as e:
            print(f"not resized, full 96x96x96 used")'''
        #gt = data
        resize = self.cfg['resize']
        gt_full = self.transform_sequence(gt_full, resize, is_train) # 250814 jubin change
        # gt_full = self.resize_sequence(gt_full, resize)
        #print(f'gt_full shape is {gt_full.shape}')
        
        B = gt_full.shape[0]

        # Flatten spatial dims for sampling
        coord = make_coord_grid(gt_full.shape[-4:], (0, 1), device=gt_full.device)
        coord = einops.repeat(coord, 'z h w t d -> b z h w t d', b=B)
        coord_flat = coord.view(B, -1, coord.shape[-1])  # (B, N, dim), N=Z*H*W*T
        
        gt = einops.rearrange(gt_full, 'b c z h w t -> b z h w t c')  # (B, Z, H, W, T, C)
        gt_flat = gt.view(B, -1, gt.shape[-1])  # (B, N, C)
    
        N = coord_flat.shape[1]
        sample_size = max(1, int(p * N))  # at least sample 1 coord
    
        indices = torch.randperm(N, device=gt.device)[:sample_size]  # (sample_size,)

        coord_sampled = coord_flat[:, indices, :]  # (B, sample_size, dim)
        gt_sampled = gt_flat[:, indices, :]       # (B, sample_size, C)

        #Jubin's AMP
        with torch.autocast(device_type="cuda", enabled=self.cfg['use_amp'], dtype=torch.bfloat16):
            pred = self.model_ddp(gt_full, coord_sampled)

            #pred = hyponet(coord, tokens) # b h w 3
            #gt = einops.rearrange(gt, 'b c z h w t -> b z h w t c')
            mses = ((pred - gt_sampled)**2).view(B, -1).mean(dim=-1)
            loss = mses.mean()
            psnr = (-10 * torch.log10(mses)).mean()

        #Old Code
        '''pred = self.model_ddp(gt_full, coord_sampled)
        mses = ((pred - gt_sampled)**2).view(B, -1).mean(dim=-1)
        loss = mses.mean()
        psnr = (-10 * torch.log10(mses)).mean()'''

        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {'loss': loss.item(), 'psnr': psnr.item()}

    def train_step(self, data):
        return self._iter_step(data, is_train=True)

    def evaluate_step(self, data):
        with torch.no_grad():
            return self._iter_step(data, is_train=False)

    def _gen_vis_result(self, tag, vislist, p=0.01):
        pred_value_range = (0, 1)
        self.model_ddp.eval()
        res = []
        for data in vislist:
            #data = {k: v.unsqueeze(0).cuda() for k, v in data.items()}
            data = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
            #gt_full = data['fmri_sequence'][0] #B, C, H, W vs B, C, D, H, W, T
            #gt_full = data.pop('fmri_sequence')
            gt_full = data.pop('fmri_sequence')
            print(f'shape of gt is {gt_full.shape}')
            #gt = data
            with torch.no_grad():
                coord = make_coord_grid(gt_full.shape[-4:], [0, 1], device=gt_full.device)
                coord = coord.unsqueeze(0)
                coord_flat = coord.view(1, -1, coord.shape[-1])  # (B, N, dim), N=Z*H*W*T

                gt = einops.rearrange(gt_full, 'c z h w t -> z h w t c')  # (B, Z, H, W, T, C)
                gt = gt.unsqueeze(0)
                gt_flat = gt.view(1, -1, gt.shape[-1])  # (B, N, C)

                N = coord_flat.shape[1]
                sample_size = max(1, int(p * N))  # at least sample 1 coord
            
                # Random indices to sample, same number for each batch element
                indices = torch.randperm(N, device=gt.device)[:sample_size]  # (sample_size,)
            
                # Index coords and gt
                coord_sampled = coord_flat[:, indices, :]  # (B, sample_size, dim)
                gt_sampled = gt_flat[:, indices, :]       # (B, sample_size, C)
                #hyponet, tokens = self.model_ddp(data)
                #pred = hyponet(coord, tokens)[0]
                print(f'shape of gt is {gt_full.unsqueeze(0).shape}')
                print(f'shape of coord is {coord_sampled.shape}')
                pred = self.model_ddp(gt_full.unsqueeze(0), coord_sampled)[0]
                pred = einops.rearrange(pred.clamp(*pred_value_range), 'z h w t c -> c z h w t')
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
            pass
            #self._gen_vis_result('vis_train_dataset', self.vislist_train)
        if hasattr(self, 'vislist_test'):
            pass
            #self._gen_vis_result('vis_test_dataset', self.vislist_test)
