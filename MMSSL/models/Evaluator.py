from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl

from models.ResnetEvalModel import ResnetEvalModel
from models.MultimodalEvalModel import MultimodalEvalModel
import models.ECGEvalModel as ECGEvalModel
import models.ECGTabularEvalModel as ECGTabularEvalModel
from models.LinearClassifier import LinearClassifier
from models.TabularEncoder import TabularEncoder
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


class Evaluator(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    if self.hparams.datatype == 'imaging':
      self.encoder_ecg = ResnetEvalModel(self.hparams)
    if hparams.attention_pool:
      hparams.global_pool = "attention_pool"
    if self.hparams.datatype == 'ecg' or self.hparams.datatype == 'multimodal':
      hparams.input_size = (hparams.input_channels, hparams.input_electrodes, hparams.time_steps)
      hparams.patch_size = (hparams.patch_height, hparams.patch_width)
      self.encoder_ecg = ECGEvalModel.__dict__[hparams.ecg_model](
        num_classes=hparams.num_classes,
        drop_path_rate=hparams.drop_path,
        checkpoint=self.hparams.checkpoint,
        global_pool=hparams.global_pool,
        patch_size=(hparams.patch_height, hparams.patch_width),
        img_size=(hparams.input_electrodes, hparams.time_steps)
        )
      self.encoder_ecg.blocks[-1].attn.forward = self._attention_forward_wrapper(self.encoder_ecg.blocks[-1].attn) # required to read out the attention map of the last layer

    if self.hparams.datatype == 'ecg_tabular':
      hparams.input_size = (hparams.input_channels, hparams.input_electrodes, hparams.time_steps)
      hparams.patch_size = (hparams.patch_height, hparams.patch_width)
      self.tabular_encoder = TabularEncoder(hparams)
      self.encoder_ecg = ECGTabularEvalModel.__dict__[hparams.ecg_model](
        num_classes=hparams.num_classes,
        drop_path_rate=hparams.drop_path,
        checkpoint=self.hparams.checkpoint,
        global_pool=hparams.global_pool,
        tabular_encoder=self.tabular_encoder,
        add_linear_to_fuse = hparams.add_linear_to_fuse,
        patch_size=(hparams.patch_height, hparams.patch_width),
        img_size=(hparams.input_electrodes, hparams.time_steps)
        )
      self.encoder_ecg.blocks[-1].attn.forward = self._attention_forward_wrapper(self.encoder_ecg.blocks[-1].attn) # required to read out the attention map of the last layer

    if self.hparams.datatype == 'imaging_and_tabular':
      self.encoder_ecg = MultimodalEvalModel(self.hparams)
    
    if self.hparams.datatype == 'tabular':
      self.encoder_ecg = TabularEncoder(hparams)
    
  
    # Classifier
    if hparams.online_classifier == "image" or self.hparams.datatype == 'imaging':
      self.classifier = LinearClassifier(in_size=2048, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat) # image
    elif self.hparams.datatype == 'ecg_tabular' and not hparams.add_linear_to_fuse:
      self.classifier = LinearClassifier(in_size=self.encoder_ecg.embed_dim + hparams.tabular_embedding_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat) # ecg
    elif self.hparams.datatype == 'ecg_tabular' and hparams.add_linear_to_fuse:
      self.classifier = LinearClassifier(in_size=self.encoder_ecg.embed_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat)
    elif self.hparams.datatype == 'tabular':
      self.classifier = LinearClassifier(in_size=self.encoder_ecg.args.tabular_embedding_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat)
    else:
      self.classifier = LinearClassifier(in_size=self.encoder_ecg.embed_dim, num_classes=self.hparams.num_classes, init_type=self.hparams.init_strat) # ecg

    # Load weights
    checkpoint = torch.load(self.hparams.checkpoint)
    
    original_args = checkpoint['hyper_parameters']
    state_dict = checkpoint['state_dict']
    state_dict_encoder = {}
    for k in list(state_dict.keys()):
      if k.startswith('classifier.'):
        state_dict_encoder[k[len('classifier.'):]] = state_dict[k]

    self.online_classifier = hparams.online_classifier

    self.acc_train = torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task='binary')
    self.acc_val = torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task='binary')
    self.acc_test = torchmetrics.Accuracy(num_classes=self.hparams.num_classes, task='binary')

    self.auc_train = torchmetrics.AUROC(num_classes=self.hparams.num_classes, task='binary')
    self.auc_val = torchmetrics.AUROC(num_classes=self.hparams.num_classes, task='binary')
    self.auc_test = torchmetrics.AUROC(num_classes=self.hparams.num_classes, task='binary')

    self.classifier_balanced_accuracy_train = torchmetrics.Accuracy(task = 'multiclass', num_classes = 2, average='macro')
    self.classifier_balanced_accuracy_val = torchmetrics.Accuracy(task = 'multiclass', num_classes = 2, average='macro')
    self.classifier_balanced_accuracy_test = torchmetrics.Accuracy(task = 'multiclass', num_classes = 2, average='macro')
    # Defines weights to be used for the classifier in case of imbalanced data
    self.weights = torch.tensor(self.hparams.weights)
    self.criterion = torch.nn.CrossEntropyLoss(weight=self.weights)
    
    self.best_val_score = 0

    print(self.encoder_ecg)

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


  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    if self.hparams.datatype == 'ecg_tabular':
      x, t, y = batch
      embedding = self.forward(x,t)
    elif self.hparams.datatype == 'imaging':
      x, y = batch
      embedding = self.forward(x)
    elif self.hparams.datatype == 'tabular':
      x, y = batch
      embedding = self.forward(x)
    else:
      x, y = batch
      embedding = self.forward(x)
    
    y_hat = self.classifier(embedding)
    #get 1-d logits for AUC 
    logits = torch.nn.functional.softmax(y_hat, dim=1)[:,1]
    self.auc_test(logits, y)
    y_hat = y_hat.argmax(dim=1)
    self.acc_test(y_hat, y)
    self.classifier_balanced_accuracy_test(y_hat, y)
    
  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_acc = self.acc_test.compute()
    test_auc = self.auc_test.compute()
    balanced_accuracy = self.classifier_balanced_accuracy_test.compute()

    self.log('test.acc', test_acc)
    self.log('test.auc', test_auc)
    self.log('test.balanced_accuracy', balanced_accuracy)

  def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    if self.hparams.datatype == 'ecg_tabular':
      y_hat = self.encoder_ecg.forward_features(x, t)
    elif self.hparams.datatype == 'imaging':
      y_hat = self.encoder_ecg.backbone(x).squeeze() 
    elif self.hparams.datatype == 'tabular':
      y_hat = self.encoder_ecg.forward(x)
    else:
      y_hat = self.encoder_ecg.forward_features(x)
    if len(y_hat.shape)==1:
      y_hat = torch.unsqueeze(y_hat, 0)
    return y_hat

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _, optimizer_idx: int) -> torch.Tensor:
    """
    Train and log.
    """
    if self.hparams.datatype == 'ecg_tabular':
      x, t, y = batch
      embedding = self.forward(x,t)
    elif self.hparams.datatype == 'tabular':
      x, y = batch
      embedding = self.forward(x)
    elif self.hparams.datatype == 'imaging':
      x, y = batch
      embedding = self.forward(x)    
    else:
      x, y = batch
      embedding = self.forward(x)
    y = y.to(torch.long)
    y_hat = self.classifier(embedding)
    loss = self.criterion(y_hat, y)
    #get logits
    logits = torch.nn.functional.softmax(y_hat, dim=1)[:,1]
    self.auc_train(logits, y)
    y_hat = y_hat.argmax(dim=1)
    self.acc_train(y_hat, y)
    balanced_accuracy = self.classifier_balanced_accuracy_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
    self.log('eval.train.acc', self.acc_train, on_epoch=True, on_step=False)
    self.log('eval.train.auc', self.auc_train, on_epoch=True, on_step=False)
    self.log('eval.train.balanced_accuracy', balanced_accuracy, on_epoch=True, on_step=False)

    return loss

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    self.classifier.eval()
    if self.hparams.datatype == 'ecg_tabular':
      x, t, y = batch
      embedding = self.forward(x,t)
    else:
      x, y = batch
      embedding = self.forward(x)
    y = y.to(torch.long)
    y_hat = self.classifier(embedding)
    loss = self.criterion(y_hat, y)
    #get logits
    logits = torch.nn.functional.softmax(y_hat, dim=1)[:,1]
    self.auc_val(logits, y)
    y_hat = y_hat.argmax(dim=1)
    self.acc_val(y_hat, y)
    balanced_accuracy = self.classifier_balanced_accuracy_val(y_hat, y)

    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)
    self.log('eval.val.acc', self.acc_val, on_epoch=True, on_step=False)
    self.log('eval.val.auc', self.auc_val, on_epoch=True, on_step=False)
    self.log('eval.val.balanced_accuracy', balanced_accuracy, on_epoch=True, on_step=False)

    self.classifier.train()
    
  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return  

    epoch_acc_val = self.acc_val.compute()
    epoch_auc_val = self.auc_val.compute()
    balanced_accuracy = self.classifier_balanced_accuracy_val.compute()
    self.log('eval.val.acc', epoch_acc_val, on_epoch=True, on_step=False, metric_attribute=self.acc_val)
    self.log('eval.val.auc', epoch_auc_val, on_epoch=True, on_step=False, metric_attribute=self.auc_val)
    self.log('eval.val.balanced_accuracy', balanced_accuracy, on_epoch=True, on_step=False, metric_attribute=self.classifier_balanced_accuracy_val)

    self.best_val_score = max(self.best_val_score, epoch_auc_val)

    self.acc_val.reset()
    self.auc_val.reset()

  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.AdamW(
      self.encoder_ecg.parameters(),
      lr=self.hparams.lr_eval,
      weight_decay=self.hparams.weight_decay_eval
    )
    classifier_optimizer = torch.optim.AdamW(self.classifier.parameters(), lr=self.hparams.lr_classifier, weight_decay=self.hparams.weight_decay_classifier)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.hparams.warmup_epochs, max_epochs=self.hparams.max_epochs)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(5/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.01)

    return (
      { # Contrastive
        "optimizer": optimizer, 
        "lr_scheduler": {"scheduler": scheduler, "monitor": 'eval.val.loss',
          "strict": False}
      },
      { # Classifier
        "optimizer": classifier_optimizer        
      }
    )