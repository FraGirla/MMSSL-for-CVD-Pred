from typing import List, Tuple
import random
import csv
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms
from torchvision.io import read_image
import copy

import utils.ecg_augmentations as augmentations


class ContrastiveECGAndTabularDataset(Dataset):
  """
  Multimodal dataset that generates multiple views of ECG and tabular data for contrastive learning.
  
  The first ECG view is always augmented. The second view is corrupted by replacing features
  with values chosen from the empirical marginal distribution of that feature.
  """
  def __init__(
      self,  
      data_path_ecg: str, ecg_random_crop: bool,
      data_path_tabular: str, corruption_rate: float, field_lengths_tabular: str, one_hot_tabular: bool,
      labels_path: str, img_size: int,
      args) -> None:
      
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if self.delete_segmentation:
      self.data_imaging = [image[0::2, ...] for image in self.data_imaging]

    self.img_size = img_size
    self.default_transform = transforms.Compose([
      transforms.Resize(size=(img_size, img_size),antialias=True),
      transforms.Lambda(lambda x : x.float())
    ])

    # ECG
    self.data_ecg = torch.load(data_path_ecg)
    self.data_ecg = [d.unsqueeze(0) for d in self.data_ecg]
    self.data_ecg = [d[:, :args.input_electrodes, :] for d in self.data_ecg]
    self.ecg_random_crop = ecg_random_crop

    # Tabular
    self.data_tabular = self.read_and_parse_csv(data_path_tabular)
    self.generate_marginal_distributions(data_path_tabular)
    self.c = corruption_rate
    self.field_lengths_tabular = torch.load(field_lengths_tabular)
    self.one_hot_tabular = one_hot_tabular

    # Classifier
    self.labels = torch.load(labels_path)

    self.args = args
  
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

  def generate_marginal_distributions(self, data_path: str) -> None:
    """
    Generates empirical marginal distribution by transposing data
    """
    data_df = pd.read_csv(data_path)
    self.marginal_distributions = data_df.transpose().values.tolist()

  def get_input_size(self) -> int:
    """
    Returns the number of fields in the table. 
    Used to set the input number of nodes in the MLP
    """
    if self.one_hot_tabular:
      return int(sum(self.field_lengths_tabular))
    else:
      return len(self.data[0])

  def corrupt(self, subject: List[float]) -> List[float]:
    """
    Creates a copy of a subject, selects the indices 
    to be corrupted (determined by hyperparam corruption_rate)
    and replaces their values with ones sampled from marginal distribution
    """
    subject = copy.deepcopy(subject)

    indices = random.sample(list(range(len(subject))), int(len(subject)*self.c)) 
    for i in indices:
      subject[i] = random.sample(self.marginal_distributions[i],k=1)[0] 
    return subject

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

  def generate_ecg_views(self, index: int) -> List[torch.Tensor]:
    """
    Generates two views of a subjects ECG. The first is always the augmented.
    """
    data = self.data_ecg[index]
    
    if self.ecg_random_crop:
      transform = augmentations.CropResizing(fixed_crop_len=self.args.time_steps, resize=False)
    else:
      transform = augmentations.CropResizing(fixed_crop_len=self.args.time_steps, start_idx=0, resize=False)
    ecg_orig = transform(data)

    ecg_aug = ecg_orig
    if random.random() < self.augmentation_rate:
      augment = transforms.Compose([augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise),
                                    augmentations.Jitter(sigma=self.args.jitter_sigma),
                                    augmentations.Rescaling(sigma=self.args.rescaling_sigma),
                                    # augmentations.CropResizing(fixed_crop_len=self.args.time_steps, resize=False),
                                    #augmentations.TimeFlip(prob=0.33),
                                    #augmentations.SignFlip(prob=0.33)
                                    ])
      ecg_aug = augment(ecg_aug)
    #else:
      #use only crop resizing
      #augment = transforms.Compose([augmentations.CropResizing(fixed_crop_len=self.args.time_steps, resize=False)])
      #ecg_aug = augment(ecg_aug)
    
    return ecg_aug, ecg_orig

  def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
    ecg_aug, ecg_orig = self.generate_ecg_views(index)

    tabular_views = [torch.tensor(self.data_tabular[index], dtype=torch.float), torch.tensor(self.corrupt(self.data_tabular[index]), dtype=torch.float)]
    if self.one_hot_tabular:
      tabular_views = [self.one_hot_encode(tv) for tv in tabular_views]

    #clone tensor with dtype long 
    label = self.labels[index].clone().detach().long()
    return ecg_aug, label, ecg_orig, tabular_views, index
    # # for unet encoder
    # return image_aug[0::2, ...], ecg_aug, label, image_orig[0::2, ...], ecg_orig, index

  def __len__(self) -> int:
    return len(self.data_ecg)