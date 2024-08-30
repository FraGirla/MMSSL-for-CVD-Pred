from typing import List, Tuple, Dict
import torch
from torch import nn
import torchvision
import torchmetrics
import pytorch_lightning as pl

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from lightly.models.modules import SimCLRProjectionHead
from utils.ntx_ent_loss_custom import NTXentLoss
from utils.clip_loss import CLIPLoss

import utils.pos_embed as pos_embed

from models.LinearClassifier import LinearClassifier
import models.ECGEncoder as ECGEncoder


class ECGSimCLR(pl.LightningModule):
  """
  Lightning module for ecg SimCLR.

  Alternates training between contrastive model and online classifier.
  """
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # ECG
    if hparams.attention_pool:
      hparams.global_pool = "attention_pool"
    self.encoder_ecg = ECGEncoder.__dict__[hparams.ecg_model](
        num_classes=hparams.num_classes,
        drop_path_rate=hparams.drop_path,
        global_pool=hparams.global_pool,
        patch_size=(hparams.patch_height, hparams.patch_width),
        img_size=(hparams.input_electrodes, hparams.time_steps))
    self.encoder_ecg.blocks[-1].attn.forward = self._attention_forward_wrapper(self.encoder_ecg.blocks[-1].attn) # required to read out the attention map of the last layer
    self.projection_head = SimCLRProjectionHead(self.encoder_ecg.embed_dim, self.hparams.embedding_dim, self.hparams.projection_dim)

    if self.hparams.ecg_pretrain_checkpoint:
      checkpoint = torch.load(hparams.ecg_pretrain_checkpoint)
      print("Load pre-trained checkpoint from: %s" % hparams.ecg_pretrain_checkpoint)
      state_dict = self.encoder_ecg.state_dict()
      try:
        checkpoint_model = checkpoint['model']
        for k in ['head.weight', 'head.bias']:
          if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
      except KeyError:
        checkpoint_model = checkpoint['state_dict']
        checkpoint_model = {k: v for k, v in checkpoint_model.items() if k.startswith('encoder_ecg') or k.startswith('projection_head')}
        checkpoint_model = {k.replace('encoder_ecg.', '').replace('projection_head.', ''): v for k, v in checkpoint_model.items()}
        for k in ['projection_head_ecg.head.weight', 'projection_head_ecg.head.bias']:
          if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

      # interpolate position embedding
      pos_embed.interpolate_pos_embed(self.encoder_ecg, checkpoint_model)

      # load pre-trained model
      msg = self.encoder_ecg.load_state_dict(checkpoint_model, strict=False)
      print(msg)

    self.criterion_train = NTXentLoss(temperature=self.hparams.temperature)
    self.criterion_val = NTXentLoss(temperature=self.hparams.temperature)

    # # "CLOCS" Kiyasseh et al. (2021) (https://arxiv.org/pdf/2005.13249.pdf)
    # self.criterion_train = CLIPLoss(temperature=self.hparams.temperature, lambda_0=self.hparams.lambda_0)
    # self.criterion_val = self.criterion_train
    
    # Defines weights to be used for the classifier in case of imbalanced data
    if not self.hparams.weights:
      self.hparams.weights = [1.0 for i in range(self.hparams.num_classes)]
    self.weights = torch.tensor(self.hparams.weights)
    self.nclasses = self.hparams.batch_size*2-1    
    # Classifier
    self.classifier = LinearClassifier(in_size=self.encoder_ecg.embed_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat)
    self.classifier_criterion = torch.nn.CrossEntropyLoss(weight=self.weights)

    self.top1_acc_train = torchmetrics.Accuracy(top_k=1, task = 'multiclass', num_classes = self.nclasses)
    self.top1_acc_val = torchmetrics.Accuracy(top_k=1, task = 'multiclass', num_classes = self.nclasses)

    self.classifier_acc_train = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted', task='binary') 
    self.classifier_acc_val = torchmetrics.Accuracy(num_classes = self.hparams.num_classes, average = 'weighted', task='binary')

    self.classifier_auc_train = torchmetrics.AUROC(num_classes = self.hparams.num_classes, task = 'binary')
    self.classifier_auc_val = torchmetrics.AUROC(num_classes = self.hparams.num_classes, task = 'binary')

    print(self.encoder_ecg)
    print(self.projection_head)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates projection of data.
    """
    x = self.encoder_ecg.forward_features(x).flatten(start_dim=1)
    z = self.projection_head(x)
    return z

  def _attention_forward_wrapper(self, attn_obj):
    """
    Modified version of def forward() of class Attention() in timm.models.vision_transformer
    """
    def my_forward(x):
        B, N, C = x.shape # C = embed_dim
        # (3, B, Heads, N, head_dim)
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # (B, Heads, N, N)
        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        # (B, Heads, N, N)
        attn_obj.attn_map = attn # this was added 

        # (B, N, Heads*head_dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x
    return my_forward

  def training_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _, optimizer_idx: int) -> torch.Tensor:
    """
    Alternates calculation of loss for training between contrastive model and online classifier.
    """
    x0, x1, y, indices = batch

    # Train contrastive model
    if optimizer_idx == 0:
      z0 = self.forward(x0)
      z1 = self.forward(x1)
      loss, logits, labels = self.criterion_train(z0, z1, indices)

      if self.nclasses > 5:
        top5_accuracy_cl = (logits.argsort(descending=True, dim=1)[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean()
        self.log("ecg.train.top5", top5_accuracy_cl, on_epoch=True, on_step=False)

      logits = logits.argmax(dim=1)

      self.top1_acc_train(logits, labels)
      
      self.log("ecg.train.loss", loss, on_epoch=True, on_step=False)
      self.log("ecg.train.top1", self.top1_acc_train, on_epoch=True, on_step=False)
    
    # Train classifier
    if optimizer_idx == 1:
      embedding = torch.squeeze(self.encoder_ecg.forward_features(x0))
      y_hat = self.classifier(embedding)
      y = y.to(torch.long)
      loss = self.classifier_criterion(y_hat, y)
      y_hat = y_hat.argmax(dim=1)

      f1_train_class = torchmetrics.F1Score(task = 'multiclass', num_classes = y.shape[0])

      self.classifier_acc_train(y_hat, y)
      self.classifier_auc_train(y_hat, y)

      self.log('classifier.train.loss', loss, on_epoch=True, on_step=False)
      self.log('classifier.train.f1', f1_train_class, on_epoch=True, on_step=False, metric_attribute = 'f1_val')
      self.log('classifier.train.accuracy', self.classifier_acc_train, on_epoch=True, on_step=False)
      self.log('classifier.train.auc', self.classifier_auc_train, on_epoch=True, on_step=False)

    return loss


  def validation_step(self, batch: Tuple[List[torch.Tensor], torch.Tensor], _) -> torch.Tensor:
    """
    Validate both contrastive model and classifier
    """
    x0, x1, y, indices = batch
    
    # Validate contrastive model
    z0 = self.forward(x0)
    z1 = self.forward(x1)
    loss, logits, labels = self.criterion_val(z0, z1, indices)

    if self.nclasses > 5:
      top5_accuracy_cl = (logits.argsort(descending=True, dim=1)[:, :5] == labels.unsqueeze(1)).any(dim=1).float().mean()
      self.log("ecg.val.top5", top5_accuracy_cl, on_epoch=True, on_step=False)


    logits = logits.argmax(dim=1)
    self.top1_acc_val(logits, labels)

    self.log("ecg.val.loss", loss, on_epoch=True, on_step=False)
    self.log("ecg.val.top1", self.top1_acc_val, on_epoch=True, on_step=False)

    # Validate classifier
    self.classifier.eval()
    embedding = torch.squeeze(self.encoder_ecg.forward_features(x0))
    y_hat = self.classifier(embedding)
    y = y.to(torch.long).to(y_hat.device)
    loss = self.classifier_criterion(y_hat, y)
    y_hat = y_hat.argmax(dim=1)
    f1_train_class = torchmetrics.F1Score(task = 'multiclass', num_classes = y.shape[0]).to(device=y_hat.device)(y_hat, y)
    self.classifier_acc_val(y_hat, y)
    self.classifier_auc_val(y_hat, y)

    self.log('classifier.val.loss', loss, on_epoch=True, on_step=False)
    self.log('classifier.val.f1', f1_train_class, on_epoch=True, on_step=False, metric_attribute = 'f1_val')
    self.log('classifier.val.accuracy', self.classifier_acc_val, on_epoch=True, on_step=False)
    self.log('classifier.val.auc', self.classifier_auc_val, on_epoch=True, on_step=False)
    self.classifier.train()

    return x0

  def validation_epoch_end(self, validation_step_outputs: List[torch.Tensor]) -> None:
    """
    Log an image from each validation step
    """
    if self.hparams.log_images:
      self.logger.log_image(key="ECG Example", images=[validation_step_outputs[0]])

  def configure_optimizers(self) -> Tuple[Dict, Dict]:
    """
    Define and return optimizer and scheduler for contrastive model and online classifier. 
    Scheduler for online classifier often disabled
    """
    optimizer = torch.optim.AdamW(
      [
        {'params': self.encoder_ecg.parameters()}, 
        {'params': self.projection_head.parameters()}
      ], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
    classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.hparams.lr_classifier, weight_decay=self.hparams.weight_decay_classifier)
    
    if self.hparams.scheduler == 'cosine':
      scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.anneal_max_epochs, eta_min=0, last_epoch=-1)
    elif self.hparams.scheduler == 'anneal':
      scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs)
    classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, patience=int(20/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr_classifier*0.0001)
    
    
    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": scheduler
      },
      { # Classifier
        "optimizer": classifier_optimizer
      }
    )