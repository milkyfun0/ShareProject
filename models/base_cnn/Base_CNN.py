"""
@File  :Base_CNN.py
@Author:CaoQiXuan
@Date  :24/2/2414:58
@Desc  :
"""
import torch
from torch import nn
from torch.nn import functional as F
from loss import calc_triple_loss
from models.base_cnn.Moudules import VisualEncoder, TextEncoder
from utils import cosine_similarity


class Base_CNN(nn.Module):
    def __init__(self, opt, writer=None):
        super(Base_CNN, self).__init__()
        self.device = None
        self.opt = opt
        self.writer = writer

        self.visual_encoder = VisualEncoder(opt=opt)
        self.image_info = self.visual_encoder.image_info

        self.text_encoder = TextEncoder(opt=opt)
        self.text_info = self.text_encoder.text_info

    def encode_image(self, image: torch.Tensor):
        return self.visual_encoder(images=image, share_info=self.image_info)

    def encode_text(self, text: torch.Tensor, attn_mask: torch.Tensor):
        return self.text_encoder(text=text, share_info=self.text_info, attn_mask=attn_mask)

    def forward(self, data_pair: dict):
        self.device = next(self.parameters()).device
        images_id, images, text_id, text, attn_mask = data_pair["images_id"].to(self.device), data_pair["images"].to(
            self.device), data_pair["text_id"].to(self.device), data_pair["text"].to(self.device), data_pair[
            "attn_mask"].to(self.device)

        batch_image, batch_text = len(images), len(text)
        image_features = self.visual_encoder(images=images, share_info=self.text_info)
        text_features = self.text_encoder(text=text, share_info=self.image_info, attn_mask=attn_mask)
        image_features = F.normalize(image_features, dim=1, p=2)
        # print("i", image_features)
        text_features = F.normalize(text_features, dim=1, p=2)
        # print("t", text_features)

        image_features = image_features.unsqueeze(dim=1).expand(-1, batch_text, -1)
        text_features = text_features.unsqueeze(dim=0).expand(batch_image, -1, -1)
        dual_sim = cosine_similarity(image_features, text_features)
        # print(dual_sim)
        total_loss = calc_triple_loss(dual_sim, self.opt["loss"]["margin"])
        # print(total_loss)
        return {
            "total loss": total_loss,
            "sim": dual_sim
        }
