"""
@File  :Base_Register.py
@Author:CaoQiXuan
@Date  :23/12/323:41
@Desc  : ID Loss
        Triple Loss plus
        CMPC Loss plus
"""
import torch
from torch import nn

import utils
from models.base_register.Modules import load_clip_from_json
from utils import cosine_similarity
from loss import calc_triple_loss, calc_cmpm_loss, calc_cls_loss, calc_contrastive_loss

"""
加入了 Register 块，并且用到了forward中 效果不太行

"""


class Base_Register(nn.Module):
    def __init__(self, opt, writer=None):
        super(Base_Register, self).__init__()
        self.device = None
        self.opt = opt
        self.writer = writer
        self.clip, self.clip_config = load_clip_from_json(
            config_path=opt["model"]["CLIP"]["config_path"],
            pre_train=opt["model"]["CLIP"]["pre_train"],
            opt=opt
        )
        self.image_global_proj = self.clip.visual.mlp
        self.text_global_proj = self.clip.text_projection
        # self.alignment = Alignment(
        #     top_k=opt["model"]["Alignment"]["top_k"],
        #     layers=opt["model"]["Alignment"]["layers"],
        #     heads=opt["model"]["Alignment"]["heads"],
        #     width=opt["model"]["dim"]
        # )

        self.classifier = nn.Sequential(
            nn.Linear(self.opt["model"]["dim"], self.opt["dataset"]["images_class"]),
            nn.Softmax()
        )  # not init_normal
        self.contrastive_query_proj = nn.Sequential(
            nn.Linear(self.opt["model"]["dim"], self.opt["model"]["dim"]),
            nn.ReLU()
        )  # not init_normal SimCLR
        self.contrastive_key_proj = nn.Sequential(
            nn.Linear(self.opt["model"]["dim"], self.opt["model"]["dim"]),
            nn.ReLU()
        )  # not init_normal SimCLR

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

        # image_tokens, text_tokens = self.clip(image=images, text=text, attn_mask=attn_mask)
        # image_global_feature = image_tokens[:, 0] @ self.image_global_proj
        # text_global_feature = text_tokens[:, 0] @ self.text_global_proj
        #
        # sim_global = cosine_similarity(image_global_feature, text_global_feature)
        #
        # global_triple_loss = calc_triple_loss(sim_global, self.opt["loss"]["margin"])
        #
        # total_loss = global_triple_loss
        sim_global = self.clip(image=images, text=text, attn_mask=attn_mask)
        total_loss = calc_triple_loss(sim_global, self.opt["loss"]["margin"])

        return {
            "total loss": total_loss,
            "sim": sim_global
        }
        # cmpm_loss = calc_cmpm_loss(image_global_feature, text_global_feature, images_id, text_id,
        #                            img_div=self.opt["dataset"]["img_div"])

        # image_local_tokens, text_local_tokens = self.alignment.forward(image_feature=image_global_feature,
        #                                                                image_tokens=image_tokens,
        #                                                                text_feature=text_global_feature,
        #                                                                text_tokens=text_tokens, atte_mask=attn_mask)
        #
        # sim_local = cosine_similarity(image_local_tokens[:, :, 0].squeeze(), text_local_tokens[:, :, 0].squeeze())
        #
        # local_triple_loss = calc_triple_loss(sim_local, self.opt["loss"]["margin"])
        # print(cmpm_loss, global_triple_loss, local_triple_loss)
