"""
    A basic trainer.

    The general procedure in run() is:
        make_datasets()
            create . train_loader, test_loader, dist_samplers
        make_model()
            create . model_ddp, model
        train()
            create . optimizer, epoch, log_buffer
            for epoch = 1 ... max_epoch:
                adjust_learning_rate()
                train_epoch()
                    train_step()
                evaluate_epoch()
                    evaluate_step()
                visualize_epoch()
                save_checkpoint()
"""

import os
import os.path as osp
import time

import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

import datasets
import models
import utils
from trainers import register

# import sys
# sys.path.append("")
# from fmri_dataloader import DataModule

@register('base_fmri_trainer_slurm')
class BaseTrainer():
    def __init__(self, cfg): # changed - No longer needs rank passed in
        print('init started')
        self.cfg = cfg
        env = cfg['env'] # change start [env setting]

        self.rank = env['rank']
        self.local_rank = env['local_rank']
        self.world_size = env['world_size']
        self.tot_gpus = self.world_size
        
        self.is_master = (self.rank == 0)
        self.distributed = (self.world_size > 1) # change end [env setting]

        # Setup log, tensorboard, wandb
        if self.is_master:
            logger, writer = utils.set_save_dir(env['save_dir'], replace=False)
            with open(osp.join(env['save_dir'], 'cfg.yaml'), 'w') as f:
                yaml.dump(cfg, f, sort_keys=False)

            self.log = logger.info

            self.enable_tb = True
            self.writer = writer

            if env['wandb_upload']:
                self.enable_wandb = True
                with open('wandb.yaml', 'r') as f:
                    wandb_cfg = yaml.load(f, Loader=yaml.FullLoader)
                os.environ['WANDB_DIR'] = env['save_dir']
                os.environ['WANDB_NAME'] = env['exp_name']
                os.environ['WANDB_API_KEY'] = wandb_cfg['api_key']
                wandb.init(project=wandb_cfg['project'], entity=wandb_cfg['entity'], config=cfg)
            else:
                self.enable_wandb = False
        else:
            self.log = lambda *args, **kwargs: None
            self.enable_tb = False
            self.enable_wandb = False

        # Setup distributed devices
        torch.cuda.set_device(self.local_rank) # change start [init distributed]
        self.device = torch.device('cuda', self.local_rank)


        cudnn.benchmark = env['cudnn']
        self.log(f'Environment setup done.')

    def run(self):
        self.make_datasets()

        if self.cfg.get('eval_model') is not None:
            model_spec = torch.load(self.cfg['eval_model'])['model']
            self.make_model(model_spec, load_sd=True)
            self.epoch = 0
            self.log_buffer = []
            self.t_data, self.t_model = 0, 0
            self.evaluate_epoch()
            #self.visualize_epoch()
            self.log(', '.join(self.log_buffer))
        else:
            self.make_model()
            self.train()

        if self.enable_tb:
            self.writer.close()
        if self.enable_wandb:
            wandb.finish()

    def make_datasets(self):
        """
            By default, train dataset performs shuffle and drop_last.
            Distributed sampler will extend the dataset with a prefix to make the length divisible by tot_gpus, samplers should be stored in .dist_samplers.

            Cfg example:

            train/test_dataset:
                name:
                args:
                loader: {batch_size: , num_workers: }
        """
        cfg = self.cfg
        fmri_data_cfg = cfg['env'].get('fmri_data_cfg')

        if fmri_data_cfg is not None:
            # --- NEW: Use DataModule if config is provided ---
            self.log(f"Initializing DataModule from config: {fmri_data_cfg}")
            data_module = datasets.make(cfg['data_module'])
            self.log("DataModule created")
            # data_module = DataModule(config_path=fmri_data_cfg)
            data_module.setup()
            
            self.train_loader = data_module.train_dataloader()
            self.log("Train loader ready")
            # Use the validation set for evaluation during training
            self.test_loader = data_module.val_dataloader() 
            self.log("Datasets and Dataloaders are ready from DataModule.")

        else:
            # --- ORIGINAL: Fallback to manual dataset creation ---
            self.log("No DataModule config found. Creating datasets manually.")
            cfg = self.cfg
            self.dist_samplers = []

            def make_distributed_loader(dataset, batch_size, num_workers, shuffle=False, drop_last=False):
                sampler = DistributedSampler(dataset, shuffle=shuffle) if self.distributed else None
                # When using a sampler, shuffle must be False
                loader_shuffle = shuffle and (sampler is None)
                
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size // self.world_size,
                    drop_last=drop_last,
                    sampler=sampler,
                    shuffle=loader_shuffle,
                    num_workers=num_workers, #num_workers // self.world_size,
                    pin_memory=True)
                return loader, sampler

            if cfg.get('train_dataset') is not None:
                train_dataset = datasets.make(cfg['train_dataset'])
                self.log(f'Train dataset: len={len(train_dataset)}')
                l_cfg = cfg['train_dataset']['loader']
                self.train_loader, train_sampler = make_distributed_loader(
                    train_dataset, l_cfg['batch_size'], l_cfg['num_workers'], shuffle=True, drop_last=True)
                if train_sampler: self.dist_samplers.append(train_sampler)

            if cfg.get('test_dataset') is not None:
                test_dataset = datasets.make(cfg['test_dataset'])
                self.log(f'Test dataset: len={len(test_dataset)}')
                l_cfg = cfg['test_dataset']['loader']
                self.test_loader, test_sampler = make_distributed_loader(
                    test_dataset, l_cfg['batch_size'], l_cfg['num_workers'], shuffle=False, drop_last=False)
                if test_sampler: self.dist_samplers.append(test_sampler)
            self.log("Datasets and Dataloaders are ready from manual setup.")

    def make_model(self, model_spec=None, load_sd=False):
        if model_spec is None:
            model_spec = self.cfg['model']
        model = models.make(model_spec, load_sd=load_sd)
        self.log(f'Model: #params={utils.compute_num_params(model)}')

        if self.distributed:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model.cuda(self.local_rank)
            model_ddp = DistributedDataParallel(model, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            model.cuda()
            model_ddp = model
        self.model = model
        self.model_ddp = model_ddp

    def train(self):
        """
            For epochs perform training, evaluation, and visualization.
            Note that ave_scalars update ignores the actual current batch_size.
        """
        cfg = self.cfg

        self.optimizer = utils.make_optimizer(self.model_ddp.parameters(), cfg['optimizer'])

        if 'start_epoch' in cfg:
            start_epoch = cfg['start_epoch']
        if 'resume_from' in cfg and cfg['resume_from'] is not None and osp.exists(cfg['resume_from']):
            self.log(f"Resuming from checkpoint: {cfg['resume_from']}")
            checkpoint = torch.load(cfg['resume_from'], map_location=self.device)
            self.model.load_state_dict(checkpoint['model']['sd'])
            self.optimizer.load_state_dict(checkpoint['optimizer']['sd'])
            start_epoch = checkpoint['epoch'] + 1
            self.log(f"Resumed at epoch {start_epoch}")
        else:
            self.log("Training from scratch")
            start_epoch = 1

        max_epoch = cfg['max_epoch']
        eval_epoch = cfg.get('eval_epoch', max_epoch + 1)
        vis_epoch = cfg.get('vis_epoch', max_epoch + 1)
        save_epoch = cfg.get('save_epoch', max_epoch + 1)
        epoch_timer = utils.EpochTimer(max_epoch)

        for epoch in range(start_epoch, max_epoch + 1):
            self.epoch = epoch
            self.log_buffer = [f'Epoch {epoch}']

            # Set the epoch for the sampler to ensure proper shuffling in DDP
            if self.distributed:
                # If using DataModule, sampler is attached to the loader
                if hasattr(self.train_loader, 'sampler') and self.train_loader.sampler is not None:
                    self.train_loader.sampler.set_epoch(epoch)
                # If using manual setup, samplers are in a separate list
                elif hasattr(self, 'dist_samplers'):
                    for sampler in self.dist_samplers:
                        sampler.set_epoch(epoch)

            self.adjust_learning_rate()

            self.t_data, self.t_model = 0, 0
            self.train_epoch()

            if epoch % eval_epoch == 0:
                self.evaluate_epoch()

            if epoch % save_epoch == 0:
                self.save_checkpoint(f'epoch-{epoch}.pth')
            self.save_checkpoint('epoch-last.pth')

            epoch_time, tot_time, est_time = epoch_timer.epoch_done()
            t_data_ratio = self.t_data / (self.t_data + self.t_model)
            self.log_buffer.append(f'{epoch_time} (d {t_data_ratio:.2f}) {tot_time}/{est_time}')
            self.log(', '.join(self.log_buffer))

    def adjust_learning_rate(self):
        base_lr = self.cfg['optimizer']['args']['lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = base_lr
        self.log_temp_scalar('lr', self.optimizer.param_groups[0]['lr'])

    def log_temp_scalar(self, k, v, t=None):
        if t is None:
            t = self.epoch
        if self.enable_tb:
            self.writer.add_scalar(k, v, global_step=t)
        if self.enable_wandb:
            wandb.log({k: v}, step=t)

    def dist_all_reduce_mean_(self, x):
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x.div_(self.tot_gpus)

    def sync_ave_scalars_(self, ave_scalars):
        for k in ave_scalars.keys():
            x = torch.tensor(ave_scalars[k].item(), dtype=torch.float32, device=self.device)
            self.dist_all_reduce_mean_(x)
            ave_scalars[k].v = x.item()
            ave_scalars[k].n *= self.tot_gpus

    def train_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        loss = self.model_ddp(data)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'loss': loss.item()}

    def train_epoch(self):
        self.model_ddp.train()
        ave_scalars = dict()

        pbar = self.train_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='train', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.train_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            if self.is_master:
                pbar.set_description(desc=f'train: loss={ret["loss"]:.4f}')
            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'train:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('train/' + k, v.item())
        self.log_buffer.append(logtext)

    def evaluate_step(self, data):
        data = {k: v.cuda() for k, v in data.items()}
        with torch.no_grad():
            loss = self.model_ddp(data)
        return {'loss': loss.item()}

    def evaluate_epoch(self):
        self.model_ddp.eval()
        ave_scalars = dict()

        pbar = self.test_loader
        if self.is_master:
            pbar = tqdm(pbar, desc='eval', leave=False)

        t1 = time.time()
        for data in pbar:
            t0 = time.time()
            self.t_data += t0 - t1
            ret = self.evaluate_step(data)
            self.t_model += time.time() - t0

            B = len(next(iter(data.values())))
            for k, v in ret.items():
                if ave_scalars.get(k) is None:
                    ave_scalars[k] = utils.Averager()
                ave_scalars[k].add(v, n=B)

            t1 = time.time()

        if self.distributed:
            self.sync_ave_scalars_(ave_scalars)

        logtext = 'eval:'
        for k, v in ave_scalars.items():
            logtext += f' {k}={v.item():.4f}'
            self.log_temp_scalar('test/' + k, v.item())
        self.log_buffer.append(logtext)

    def visualize_epoch(self):
        pass

    def save_checkpoint(self, filename):
        if not self.is_master:
            return
        model_spec = self.cfg['model']
        model_spec['sd'] = self.model.state_dict()
        optimizer_spec = self.cfg['optimizer']
        optimizer_spec['sd'] = self.optimizer.state_dict()
        checkpoint = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': self.epoch,
            'cfg': self.cfg,
        }
        torch.save(checkpoint, osp.join(self.cfg['env']['save_dir'], filename))