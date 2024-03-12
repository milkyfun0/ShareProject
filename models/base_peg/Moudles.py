#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:59
# @Author  : CaoQixuan
# @File    : Moudles.py
# @Description :
import json
import math
from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn

from Collection import VarCollection


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        # print(x)
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim=-1, eps=1e-8):
    """
    :param x1:  len(x1), len(x2), dim
    :param x2:  len(x1), len(x2), dim
    :param dim:
    :param eps:
    :return: cosine similarity between x1 and x2, computed along dim.
    """

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    @VarCollection("att_weight")
    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x, att_weight = self.attn(x, x, x, need_weights=True, key_padding_mask=attn_mask, average_attn_weights=False)
        return x  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, writer=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for i in range(self.layers):
            x = self.resblocks[i](x, attn_mask)
        return x


class CrossTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width

class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, stride=2):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim))
        self.stride = stride
        self.ln_1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            LayerNorm(embed_dim)
        )

    def tokens_attention(self, tokens):
        att = torch.mean(tokens, dim=2, keepdim=True)
        tokens = tokens * torch.sigmoid(att)
        return self.mlp(tokens)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]

        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        tokens = self.proj(cnn_feat).flatten(2).transpose(1, 2)
        tokens = self.ln_1(tokens)
        tokens = tokens + self.tokens_attention(tokens)

        x = torch.cat((cls_token, tokens), dim=1)
        x = x.permute(1, 0, 2)
        return x


class PyramidTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, writer=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList()
        self.down_blocks = nn.ModuleList()
        for layer in range(layers):
            self.down_blocks.append(PosCNN(in_chans=width, embed_dim=width))
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask=attn_mask))
            self.resblocks.append(ResidualAttentionBlock(width, heads, attn_mask=attn_mask))

    def forward(self, x: torch.Tensor):
        for i in range(self.layers):
            x = self.down_blocks[i](x)
            x = self.resblocks[i](x)
            x = self.resblocks[i + 1](x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 writer=None):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size,
                               bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers // 2, heads)
        self.high_transformer = PyramidTransformer(width, layers // 6, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.iter = 0
        self.writer = writer

    def patch_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid] # 将图片分成[32,32]个patch [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ，更换位置 [1,49,768]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                            device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,50,768]
        x = x + self.positional_embedding.to(x.dtype)  # 这里位置编码是可学习的参数，可能是切了path顺序让模型自己学习吧  [1,50,768]
        x = self.ln_pre(x)  # [1,50,768]
        return x

    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)

        x = x.permute(1, 0, 2)  # NLD -> LND  # [pixel,b,d_model]=[50,1,768]
        # 当实例化时 batch——first默认维false
        x = self.transformer(x)  # 多头transformer [50,1,768]
        x = self.high_transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,50,768]
        # if self.training:
        #     utils.mark_corr_matrix(x, self.iter, writer=self.writer)
        #     self.iter += 1
        x = self.ln_post(x)  # 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]
        # x = torch.mean(x, dim=1, keepdim=False)
        x = x[:, 0, :]

        if self.proj is not None:  # self.proj是可学习参数，维度为[768,512]
            x = x @ self.proj  # 通过学习参数将维度再次融合变成512特征，最终为[1,512]

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 writer=None,
                 ):
        super().__init__()

        self.context_length = context_length  # 77
        self.Eiters = 0
        self.writer = writer
        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            writer=writer
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)  #
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, attn_mask: Optional[torch.Tensor] = None):
        # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
        x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
        x = self.transformer(x, attn_mask)  # 共11个 和图像encode结构一致 [77,3,512]
        x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
        x = self.ln_final(x).type(self.dtype)
        x = x[:, 0, :] @ self.text_projection

        return x

    def forward(self, image: torch.Tensor, text, attn_mask):

        attn_mask = (attn_mask == 0)
        batch_image, batch_text = len(image), len(text)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attn_mask)

        image_features = image_features.unsqueeze(dim=1).expand(-1, batch_text, -1)
        text_features = text_features.unsqueeze(dim=0).expand(batch_image, -1, -1)

        dual_sim = cosine_similarity(image_features, text_features)

        return dual_sim


def trans_state_dict(model_state_dict: OrderedDict, dict_path: str):
    """
    这里主要是解决TinyCLIP和CLIP参数字典之间的差距
    :param model_state_dict: 模型的state_dict
    :param dict_path: TinyCLIP的字典路径
    :param store_path:
    :return: OrderedDict 返回符合model格式的字典
    """
    state_dict = torch.load(dict_path)["state_dict"]
    new_dict = OrderedDict()
    for key in state_dict.keys():
        if "module" in key:
            if key[7:] in model_state_dict.keys():  # 源TinyCLIP中加入了modules，这里去除
                new_dict[key[7:]] = state_dict[key]
            else:
                print(key, end=" ")
        else:
            new_dict[key] = state_dict[key]
            print(key)
    print("  is not load ! ")
    torch.save(new_dict, dict_path)
    return new_dict


def get_freeze_layers_names():
    names = [
        "positional_embedding",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.conv1.weight",
        "visual.conv1.bias",
    ]
    for i in range(6):
        names.append("visual.transformer.resblocks.{}.attn.in_proj_weight".format(i))
        names.append("visual.transformer.resblocks.{}.attn.in_proj_bias".format(i))
        names.append("visual.transformer.resblocks.{}.attn.out_proj.weight".format(i))
        names.append("visual.transformer.resblocks.{}.attn.out_proj.bias".format(i))
        names.append("visual.transformer.resblocks.{}.ln_1.weight".format(i))
        names.append("visual.transformer.resblocks.{}.ln_1.bias".format(i))
        names.append("visual.transformer.resblocks.{}.mlp.c_fc.weight".format(i))
        names.append("visual.transformer.resblocks.{}.mlp.c_fc.bias".format(i))
        names.append("visual.transformer.resblocks.{}.mlp.c_proj.weight".format(i))
        names.append("visual.transformer.resblocks.{}.mlp.c_proj.bias".format(i))
        names.append("visual.transformer.resblocks.{}.ln_2.weight".format(i))
        names.append("visual.transformer.resblocks.{}.ln_2.bias".format(i))

    for i in range(3):
        names.append("transformer.resblocks.{}.attn.in_proj_weight".format(i))
        names.append("transformer.resblocks.{}.attn.in_proj_bias".format(i))
        names.append("transformer.resblocks.{}.attn.out_proj.weight".format(i))
        names.append("transformer.resblocks.{}.attn.out_proj.bias".format(i))
        names.append("transformer.resblocks.{}.ln_1.weight".format(i))
        names.append("transformer.resblocks.{}.ln_1.bias".format(i))
        names.append("transformer.resblocks.{}.mlp.c_fc.weight".format(i))
        names.append("transformer.resblocks.{}.mlp.c_fc.bias".format(i))
        names.append("transformer.resblocks.{}.mlp.c_proj.weight".format(i))
        names.append("transformer.resblocks.{}.mlp.c_proj.bias".format(i))
        names.append("transformer.resblocks.{}.ln_2.weight".format(i))
        names.append("transformer.resblocks.{}.ln_2.bias".format(i))
    return names


def special_transformer_dict(state_dict: dict):
    new_dict = {}
    for k, v in state_dict.items():
        flag = 0
        for i in range(4):
            if "visual.transformer.resblocks.{}".format(6 + i) in k:
                new_k = k.replace("visual.transformer.resblocks.{}".format(6 + i),
                                  "visual.high_transformer.resblocks.{}".format(i))
                new_dict[new_k] = v
                flag += 1
        if flag == 0:
            new_dict[k] = v
    return new_dict


def trans_standard_static_dict(state_dict: dict):
    new_dict = {}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for k, v in state_dict.items():
        if "_image_encoder" in k:
            new_dict[k[22:]] = v
        elif "_text_encoder.module." in k:
            new_dict[k[21:]] = v
        elif "_logit_scale.module." in k:
            new_dict[k[20:]] = v
        elif "module." in k:
            new_dict[k[7:]] = v
        else:
            new_dict[k] = v
    new_dict = special_transformer_dict(new_dict)
    return new_dict


def load_clip_from_json(config_path: str, pre_train: bool = True, writer=None):
    # print(config_path)
    with open(config_path + "config.json", 'r') as f:
        model_cfg = json.load(f)
    model = CLIP(
        embed_dim=model_cfg["embed_dim"],
        image_resolution=model_cfg["vision_cfg"]["image_size"],
        vision_layers=model_cfg["vision_cfg"]["layers"],
        vision_width=model_cfg["vision_cfg"]["width"],
        vision_patch_size=model_cfg["vision_cfg"]["patch_size"],

        context_length=model_cfg["text_cfg"]["context_length"],
        vocab_size=model_cfg["text_cfg"]["vocab_size"],
        transformer_width=model_cfg["text_cfg"]["width"],
        transformer_heads=model_cfg["text_cfg"]["heads"],
        transformer_layers=model_cfg["text_cfg"]["layers"],
        writer=writer
    )

    # for name, param in model.named_parameters():
    #     print(name)
    if pre_train:
        weights_map = trans_standard_static_dict(torch.load(config_path + "/" + model_cfg["name"]))
        # print(weights_map.keys())
        model.load_state_dict(weights_map, strict=False)

    # freeze_names = get_freeze_layers_names()
    # for name, param in model.named_parameters():
    #     if name in freeze_names:
    #         param.requires_grad = False

    return model, model_cfg
