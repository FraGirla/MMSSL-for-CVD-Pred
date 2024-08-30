from typing import List, Tuple
import random
import csv

import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import utils.ecg_augmentations as augmentations

class EvalECGTabularDataset(Dataset):
  """"
  Dataset for the evaluation of ECG-Tabular data
  """
  def __init__(self, data_path: str, labels_path: str, augmentation_rate: float, 
              data_path_tabular: str, field_lengths_tabular: str, eval_one_hot: bool, train: bool, args):
    super(EvalECGTabularDataset, self).__init__()
    self.data = torch.load(data_path)
    self.data = [d.unsqueeze(0) for d in self.data]
    self.data = [d[:, :args.input_electrodes, :] for d in self.data]
    
    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    #self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.eval_one_hot = eval_one_hot
    
    self.labels = torch.load(labels_path)
    self.labels = self.labels.to(torch.long)

    print(f"Data length: {len(self.data)}")
    print(f"Labels length: {len(self.labels)}")
    print(f"Tabular data length: {len(self.data_tabular)}")

    self.augmentation_rate = augmentation_rate
    self.train = train
    self.args = args

    self.transform_train = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=self.args.time_steps, resize=False),
      augmentations.FTSurrogate(phase_noise_magnitude=args.ft_surr_phase_noise),
      augmentations.Jitter(sigma=args.jitter_sigma),
      augmentations.Rescaling(sigma=args.rescaling_sigma),
      #augmentations.TimeFlip(prob=0.5),
      #augmentations.SignFlip(prob=0.5),
      #augmentations.SpecAugment(masking_ratio=0.25, n_fft=120)
    ])

    self.transform_val = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
    ])

    self.eval_train_augment_rate = args.eval_train_augment_rate

  def read_and_parse_csv(self, path_tabular: str) -> List[List[float]]:
    """
    Does what it says on the box.
    """
    with open(path_tabular,'r') as f:
      reader = csv.reader(f)
      data = []
      for r in reader:
        r2 = [float(r1) for r1 in r]
        data.append(r2)
    return data

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.eval_one_hot:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def one_hot_encode(self, subject: torch.Tensor) -> torch.Tensor:
    """
    One-hot encodes a subject's features
    """
    out = []
    for i in range(len(subject)):
      if self.field_lengths_tabular[i] == 1:
        out.append(subject[i].unsqueeze(0))
      else:
        out.append(torch.nn.functional.one_hot(subject[i].long(), num_classes=int(self.field_lengths_tabular[i])))
    return torch.cat(out)

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    data, label = self.data[index], self.labels[index]

    if self.train and (random.random() <= self.eval_train_augment_rate):
      data = self.transform_train(data)
    else:
      data = self.transform_val(data)
    
    if self.eval_one_hot:
      tab = self.one_hot_encode(torch.tensor(self.data_tabular[index]))
    else:
      tab = torch.tensor(self.data_tabular[index], dtype=torch.float)

    return data, tab, label