#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/9 14:47
# @Author  : CaoQixuan
# @File    : Base_Peg.py
# @Description :
import torch
from torch import nn

import utils
from loss import calc_triple_loss, calc_cls_loss, calc_contrastive_loss
from models.base_peg.Moudles import load_clip_from_json
from utils import cosine_similarity


class Base_Peg(nn.Module):
    def __init__(self, opt, writer=None):
        super(Base_Peg, self).__init__()
        self.device = None
        self.opt = opt
        self.writer = writer
        self.clip, self.clip_config = load_clip_from_json(
            config_path=opt["model"]["CLIP"]["config_path"],
            pre_train=opt["model"]["CLIP"]["pre_train"],
            writer=writer
        )

    def show(self):
        print("--- Network ---")
        total = utils.params_count(self)
        print('Model has {} parameters'.format(total))
        image = utils.params_count(self.clip.visual)
        text = utils.params_count(self.clip.transformer)
        print('Image Extract  has {} parameters  {:.1%}'.format(image, image / total))
        print('Text Extract has {} parameters {:.1%}'.format(text, text / total))

    def encode_image(self, image: torch.Tensor):
        return self.clip.encode_image(image)

    def encode_text(self, text: torch.Tensor, attn_mask: torch.Tensor):
        return self.clip.encode_text(text=text, attn_mask=attn_mask == 0)

    def validate_forward(self, data_pair: dict):
        """跨模态 专用 validate 函数，不进行loss 计算"""
        self.device = next(self.parameters()).device
        images_id, images, images_class, text_id, text, attn_mask = data_pair["images_id"].to(self.device), data_pair[
            "images"].to(
            self.device), data_pair["images_class"].to(self.device), data_pair["text_id"].to(self.device), data_pair[
            "text"].to(self.device), data_pair[
            "attn_mask"].to(self.device)

        sim = cosine_similarity(self.encode_image(images), self.encode_text(text, attn_mask))

        return {
            "sim": sim
        }

    def classify_loss(self, feature: torch.Tensor, label: torch.Tensor):
        prob = self.classifier(feature)
        classify_loss = calc_cls_loss(logistic=prob, labels=label)
        return classify_loss

    def contrastive_loss(self, query: torch.Tensor, query_label: torch.Tensor, key: torch.Tensor = None,
                         key_label: torch.Tensor = None):
        query = self.contrastive_query_proj(query)
        key = self.contrastive_key_proj(key) if key is not None else None
        if key is None:
            loss = calc_contrastive_loss(query, query_label, query, query_label, mask_diag=True,
                                         t=self.opt["loss"]["T"])
        else:
            loss = calc_contrastive_loss(query, query_label, key, key_label, mask_diag=False, t=self.opt["loss"]["T"])

        return loss

    def forward(self, data_pair: dict):
        self.device = next(self.parameters()).device
        images_id, images, text_id, text, attn_mask = data_pair["images_id"].to(self.device), data_pair["images"].to(
            self.device), data_pair["text_id"].to(self.device), data_pair["text"].to(self.device), data_pair[
            "attn_mask"].to(self.device)

        sim_global = self.clip(image=images, text=text, attn_mask=attn_mask)
        total_loss = calc_triple_loss(sim_global, self.opt["loss"]["margin"])

        return {
            "total loss": total_loss,
            "sim": sim_global
        }
