import os 
import sys
import time
import random

import hydra
from omegaconf import DictConfig, open_dict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from trainers.train_imaging import train_imaging
from trainers.train_ecg import train_ecg
from trainers.train_multimodal import train_multimodal
from trainers.train_trimodal import train_trimodal
from trainers.evaluate import evaluate
from trainers.test import test
from utils.utils import grab_arg_from_checkpoint, prepend_paths, chkpt_contains_arg

torch.multiprocessing.set_sharing_strategy('file_system')
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

@hydra.main(config_path='./configs', config_name='config', version_base=None)
def run(args: DictConfig):
  pl.seed_everything(args.seed)
  args = prepend_paths(args)

  print(torch.__version__)
  print(torch.cuda.is_available())
  print(torch.version.cuda)
  #print(torch.backends.cudnn.version())

  time.sleep(random.randint(1,10)) # Prevents multiple runs getting the same version

  if args.resume_training:
    wandb_id = args.wandb_id
    checkpoint = args.checkpoint
    ckpt = torch.load(args.checkpoint)
    args = ckpt['hyper_parameters']
    with open_dict(args):
      args.checkpoint = checkpoint
      args.resume_training = True
      if not wandb_id in args or not args.wandb_id:
        args.wandb_id = wandb_id
  
  
  base_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
  if args.use_wandb:
    if args.test or args.test_and_eval:
      wandb_logger = WandbLogger(project='Trimodal Eval and Test', offline=args.offline)
    elif args.task == 'regression':
      wandb_logger = WandbLogger(project='Trimodal Regression', offline=args.offline)
    elif args.run_imaging:
      wandb_logger = WandbLogger(project='SSL_imaging', save_dir=base_dir, offline=args.offline)
      args.wandb_id = wandb_logger.version
    elif args.run_ecg:
      wandb_logger = WandbLogger(project='SSL_ecg', save_dir=base_dir, offline=args.offline)
      args.wandb_id = wandb_logger.version
    elif args.run_multimodal:
      wandb_logger = WandbLogger(project='MMCL', save_dir=base_dir, offline=args.offline)
      args.wandb_id = wandb_logger.version
    elif args.run_trimodal:
      wandb_logger = WandbLogger(project='Trimodal', save_dir=base_dir, offline=args.offline)
      args.wandb_id = wandb_logger.version
    else:
      raise ValueError('No method selected')
  else:
    wandb_logger = None

  checkpoints = []
  datasets = []

  if args.run_imaging:
    args.datatype = 'imaging'
    version = train_imaging(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'imaging', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('imaging')
  
  if args.run_ecg:
    args.datatype = 'ecg'
    version = train_ecg(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'ecg', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('ecg')

  if args.run_multimodal:
    args.datatype = 'multimodal'
    version = train_multimodal(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'multimodal', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('multimodal')

  if args.run_trimodal:
    args.datatype = 'trimodal'
    version = train_trimodal(args, wandb_logger)
    checkpoint = os.path.join(base_dir, 'runs', 'trimodal', f'version_{version}', 'checkpoint_best_loss.ckpt')
    checkpoints.append(checkpoint)
    datasets.append('trimodal')
  
  for i in range(len(checkpoints)):
     args.checkpoint = checkpoints[i]
     args.datatype = datasets[i]
     if args.test:
       test(args, wandb_logger)
     else:
       evaluate(args, wandb_logger)

if __name__ == "__main__":
  run()
