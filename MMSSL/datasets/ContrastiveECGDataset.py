from typing import Any, Tuple

import random

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import utils.ecg_augmentations as augmentations


class ContrastiveECGDataset(Dataset):
  """Fast EEGDataset (fetching prepared data and labels from files)"""
  def __init__(self, data_path: str, labels_path: str, transform=None, augmentation_rate: float=1.0, args=None, train = None) -> None:
    """
    data_path:            Path to torch file containing images
    labels_path:          Path to torch file containing labels
    transform:            Compiled torchvision augmentations
    """
    self.transform = transform
    self.augmentation_rate = augmentation_rate

    self.default_transform = transforms.Compose([
      augmentations.CropResizing(fixed_crop_len=args.input_size[-1], resize=False)
    ])

    # # "CLOCS" Kiyasseh et al. (2021) (https://arxiv.org/pdf/2005.13249.pdf)
    # self.default_transform_1 = transforms.Compose([
    #   augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
    # ])
    # self.default_transform_2 = transforms.Compose([
    #   augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
    # ])

    self.data_ecg = torch.load(data_path) # load to ram
    self.data_ecg = [d.unsqueeze(0) for d in self.data_ecg]
    self.data_ecg = [d[:, :args.input_electrodes, :] for d in self.data_ecg]

    self.labels = torch.load(labels_path) # load to ram

    print(f"Loaded {len(self.data_ecg)} samples and {len(self.labels)} labels")
    print(f"Positive samples: {sum(self.labels)}, Negative samples: {len(self.labels) - sum(self.labels)}")
    
    if train:
      #get indices of positive samples
      self.positive_indices = [i for i, label in enumerate(self.labels) if label == 1]
      #get indices of negative samples
      self.negative_indices = [i for i, label in enumerate(self.labels) if label == 0]

      #make a dataset composed of both all positive samples and random negative samples (with same size)
      self.indices = self.positive_indices + random.sample(self.negative_indices, len(self.positive_indices))
      self.data_ecg = [self.data_ecg[i] for i in self.indices]

      self.labels = [self.labels[i] for i in self.indices]

  def __len__(self) -> int:
    """
    return the number of samples in the dataset
    """
    
    return len(self.labels)

  def __getitem__(self, idx) -> Tuple[Any, Any]:
    """
    returns two augmented views of one signal and its label
    """
    data = self.data_ecg[idx]
    
    view_1 = self.transform(self.default_transform(data))
    view_2 = self.default_transform(data)
    if random.random() < self.augmentation_rate:
      view_2 = self.transform(view_2)
  
    return view_1, view_2, self.labels[idx], idx