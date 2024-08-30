from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from datasets.EvalImageDataset import EvalImageDataset
from datasets.EvalECGDataset import EvalECGDataset
from datasets.EvalECGTabularDataset import EvalECGTabularDataset
from datasets.EvalTabularDataset import EvalTabularDataset
from models.Evaluator import Evaluator
from utils.utils import grab_arg_from_checkpoint
import torch

def test(hparams, wandb_logger=None):
  """
  Tests trained models. 
  
  IN
  hparams:      All hyperparameters
  """
  pl.seed_everything(hparams.seed)
  
  if hparams.datatype == 'imaging' or hparams.datatype == 'multimodal':
    test_dataset = EvalImageDataset(hparams.data_test_eval_imaging, hparams.labels_test_eval_imaging, hparams.delete_segmentation, 0, grab_arg_from_checkpoint(hparams.checkpoint, 'img_size'), target=hparams.target, train=False, live_loading=hparams.live_loading, task = 'classification')
    print(test_dataset.transform_val.__repr__())
  elif hparams.datatype == 'tabular':
      test_dataset = EvalTabularDataset(hparams.data_test_eval_tabular, hparams.labels_test_eval_tabular, hparams.eval_one_hot, hparams.field_lengths_tabular)
      hparams.input_size = test_dataset.get_input_size()
  elif hparams.datatype == 'ecg':
    test_dataset = EvalECGDataset(hparams.data_test_eval_ecg, hparams.labels_test_eval_ecg, hparams.eval_train_augment_rate, train=False, args=hparams)
  elif hparams.datatype == 'ecg_tabular':
    test_dataset = EvalECGTabularDataset(hparams.data_test_eval_ecg_trimodal, hparams.labels_test_eval_ecg_trimodal, hparams.eval_train_augment_rate, 
        hparams.data_test_eval_tabular, hparams.field_lengths_tabular, hparams.eval_one_hot, train=False, args=hparams)
  else:
    raise Exception('argument dataset must be set to imaging, tabular or multimodal')
  
  drop = ((len(test_dataset)%hparams.batch_size)==1)

  test_loader = DataLoader(
    test_dataset,
    num_workers=hparams.num_workers, batch_size=hparams.batch_size,  
    pin_memory=True, shuffle=False, persistent_workers=True, drop_last=drop)

  hparams.dataset_length = len(test_loader)

  model = Evaluator(hparams)
  model.freeze()
  trainer = Trainer.from_argparse_args(hparams, accelerator='gpu', devices=1, logger=wandb_logger)

  #load model
  checkpoint = torch.load(hparams.checkpoint)
  original_args = checkpoint['hyper_parameters']
  state_dict = checkpoint['state_dict']
  log = model.load_state_dict(state_dict=state_dict, strict=False)
  print("Loaded weights from checkpoint")
  print(log)
  trainer.test(model, test_loader)