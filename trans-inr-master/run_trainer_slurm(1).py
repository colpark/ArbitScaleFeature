"""
    Generate a cfg object according to a cfg file and args, then spawn Trainer(rank, cfg).
"""
import argparse
import os
import yaml
import torch
import utils
import trainers

def parse_args(): # changed - remove port-offset since we are using environment variable now
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True)
    parser.add_argument('--fmri-data-cfg', required=False, help="Path to the DataModule's config YAML.")
    parser.add_argument('--load-root', default='data')
    parser.add_argument('--save-root', default='save')
    parser.add_argument('--name', '-n', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--cudnn', action='store_true')
    parser.add_argument('--wandb-upload', '-w', action='store_true')
    args = parser.parse_args()
    return args

def make_cfg(args):
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    def translate_cfg_(d):
        for k, v in d.items():
            if isinstance(v, dict):
                translate_cfg_(v)
            elif isinstance(v, str):
                d[k] = v.replace('$load_root$', args.load_root)
    translate_cfg_(cfg)

    if args.name is None:
        exp_name = os.path.basename(args.cfg).split('.')[0]
    else:
        exp_name = args.name
    if args.tag is not None:
        exp_name += '_' + args.tag

    env = dict() # changed - remove tot_gpus & port since we are using environment variables
    env['exp_name'] = exp_name
    env['save_dir'] = os.path.join(args.save_root, exp_name)
    env['cudnn'] = args.cudnn
    env['wandb_upload'] = args.wandb_upload
    if args.fmri_data_cfg:
        env['fmri_data_cfg'] = args.fmri_data_cfg
    cfg['env'] = env

    return cfg

def main(): # changed - we use os.environ to manage distributed training things
    """
    Main entry point for a single Slurm task.
    Reads DDP configuration from environment variables.
    """
    args = parse_args()
    cfg = make_cfg(args)
    
    # Get DDP info from Slurm environment variables
    cfg['env']['rank'] = int(os.environ['RANK'])
    cfg['env']['local_rank'] = int(os.environ['LOCAL_RANK'])
    cfg['env']['world_size'] = int(os.environ['WORLD_SIZE'])

    # Only allow the master process (rank 0) to handle directory setup
    if cfg['env']['rank'] == 0:
        utils.ensure_path(cfg['env']['save_dir'])

    # Use a barrier to make all other processes wait until rank 0 is done
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        torch.distributed.barrier()
    
    # Create and run the trainer
    trainer = trainers.trainers_dict[cfg['trainer']](cfg)
    trainer.run()

if __name__ == '__main__':
    main()