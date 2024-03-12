"""
@Author:CaoQiXuan
@Date  :23/12/323:41
@Desc  : ID Loss
        Triple Loss plus
        CMPC Loss plus
"""
import torch
from torch import nn

import utils
from models.base_clip.CLIP import load_clip_from_json, FineAlign
from models.base_clip.Alignment import Alignment
from utils import cosine_similarity
from loss import calc_triple_loss, calc_cmpm_loss, calc_cls_loss, calc_contrastive_loss, calc_mask_loss


class Network(nn.Module):
    def __init__(self, opt, writer=None):
        super(Network, self).__init__()
        self.device = None
        self.opt = opt
        self.writer = writer
        self.clip, self.clip_config = load_clip_from_json(
            config_path=opt["model"]["CLIP"]["config_path"],
            pre_train=opt["model"]["CLIP"]["pre_train"],
            writer=writer
        )
        # self.align = FineAlign(
        #     width=self.opt["model"]["dim"],
        #     layers=1,
        #     vocab_size=self.clip_config["text_cfg"]["vocab_size"])

    def show(self):
        print("--- Network ---")
        total = utils.params_count(self)
        print('Model has {} parameters'.format(total))
        image = utils.params_count(self.clip.visual)
        text = utils.params_count(self.clip.transformer)

        print('Image Extract  has {} parameters  {:.1%}'.format(image, image / total))
        print('Text Extract has {} parameters {:.1%}'.format(text, text / total))
        # align = utils.params_count(self.align)
        # if self.align is not None:
        #     print('Align Extract has {} parameters {:.1%}'.format(align, align / total))

    def encode_image(self, image: torch.Tensor):
        return self.clip.encode_image(image)[0]

    def encode_text(self, text: torch.Tensor, attn_mask: torch.Tensor):
        return self.clip.encode_text(text=text, attn_mask=attn_mask == 0)[0]

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

    def mask_loss(self, feature: torch.Tensor, label: torch.Tensor):
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
        images_id, images, text_id, text, attn_mask, mask_caption, mask_label = data_pair["images_id"].to(
            self.device), data_pair["images"].to(self.device), data_pair["text_id"].to(
            self.device), data_pair["text"].to(self.device), data_pair["attn_mask"].to(
            self.device), data_pair["mask_caption"].to(self.device), data_pair["mask_label"].to(self.device)

        sim_global, image_tokens, text_tokens = self.clip(image=images, text=text, attn_mask=attn_mask)

        # scores = self.align(text_tokens, image_tokens, attn_mask == 0).reshape(
        #     -1, self.clip_config["text_cfg"]["vocab_size"])
        # labels = mask_label.reshape(-1)
        align_loss = 0

        # calc_contra = calc_contrastive_loss(image_feature, text_feature, images_id, text_id)
        triple_loss = calc_triple_loss(sim_global, self.opt["loss"]["margin"])
        # print(align_loss, triple_loss)
        total_loss = align_loss + triple_loss
        # print(calc_contra, triple_loss)
        return {
            "total loss": total_loss,
            "sim": sim_global
        }
