# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed, Block


class ECGTabularEvalModel(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, checkpoint: str, tabular_encoder = None, global_pool: bool = False, add_linear_to_fuse = None, **kwargs):
        super(ECGTabularEvalModel, self).__init__(**kwargs)
        #TODO
        #embed_dim = 384
        #patch_size = (1, 100)
        #img_size = (12, 2500)
        self.embed_dim = kwargs['embed_dim']
        self.patch_size = kwargs['patch_size']
        self.img_size = kwargs['img_size']
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, embed_dim=self.embed_dim, in_chans=1)
        
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.global_pool = global_pool
        if self.global_pool == "attention_pool":
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'], num_heads=kwargs['num_heads'], batch_first=True)
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            # del self.norm  # remove the original norm

        # Tabular
        self.tabular_encoder = tabular_encoder
        if add_linear_to_fuse:
            self.has_head = True
            self.head = nn.Linear(2*embed_dim, embed_dim)
        else:
            self.has_head = False

        # Load weights
        checkpoint = torch.load(checkpoint)

        original_args = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']
        state_dict_encoder = {}
        for k in list(state_dict.keys()):
          if k.startswith('encoder_ecg.'):
            state_dict_encoder[k[len('encoder_ecg.'):]] = state_dict[k]

        log = self.load_state_dict(state_dict_encoder, strict=False)
        print("Loaded weights from checkpoint in ECGTabularEvalModel")
        print(log)

    def forward_features(self, x, t, localized=False):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if localized:
            outcome = x[:, 1:]
        elif self.global_pool == "attention_pool":
            q = x[:, 1:, :].mean(dim=1, keepdim=True)
            k = x[:, 1:, :]
            v = x[:, 1:, :]
            x, x_weights = self.attention_pool(q, k, v) # attention pool without cls token
            outcome = self.fc_norm(x.squeeze(dim=1))
        elif self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        # forward tabular
        tabular = self.tabular_encoder(t)
        #concat
        outcome = torch.cat((outcome, tabular), dim=1)
        if self.has_head:
            outcome = self.head(outcome)
        return outcome


def vit_pluto_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=256, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_tiny_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=384, depth=3, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=512, depth=4, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_medium_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=640, depth=6, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_big_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=768, depth=8, num_heads=8, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base_patch200(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 200), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch100(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 100), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch50(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 50), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch10(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 10), embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch224(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 224), embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch112(**kwargs):
    model = ECGTabularEvalModel(
        patch_size=(65, 112), embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patchX(**kwargs):
    model = ECGTabularEvalModel(
        embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model