"""
@File  :Alignment.py
@Author:CaoQiXuan
@Date  :23/12/120:13
@Desc  : 细粒度对齐
"""
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import LayerNorm

from models.base_clip.CLIP import QuickGELU


def t1(image_tokens, text_tokens):
    """
    判断计算相似度乘法是否相同
    :param image_tokens:  b * patch * dim
    :param text_tokens:  b * s * dim
    :return:
    """
    batch_image, patch, _ = image_tokens.shape
    batch_text, s, _ = text_tokens.shape
    sim = torch.zeros((batch_image, batch_text, patch, s))
    for i, image_token in enumerate(image_tokens):
        for j, text_token in enumerate(text_tokens):
            sim[i][j] = image_token @ text_token.transpose(0, 1)
    # # test_sim
    # sim_test = t1(image_tokens, text_tokens)
    # print(torch.sum((torch.abs(sim_i2t - sim_test) > 1e-7).to(torch.int32).flatten()))
    return sim


def t2(image_tokens, image_token_mask):
    """
    测试masked_select是否工作成功
    :param image_tokens:b * b* patch * dim
    :param image_token_mask:b * b* patch * dim
    :return:
    """
    l = []
    for i, image_token in enumerate(image_tokens):
        for j, text_token in enumerate(image_token):
            for k in range(len(image_token_mask[i][j])):
                if image_token_mask[i][j][k] == 1:
                    l.append(text_token[k])
    # # test select 代码中的使用方法
    # print(torch.sum((torch.abs(image_tokens - test_image_token) > 1e-7).to(torch.int32).flatten()))
    return torch.stack(l, dim=0)


class CrossAttentionBLock(nn.Module):

    def __init__(self, d_model: int, n_head: int):
        super(CrossAttentionBLock, self).__init__()
        self.attn = nn.MultiheadAttention(d_model, d_model // 64, batch_first=True)  # n_head 头，d_model 表示维度。
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.ln_3 = LayerNorm(d_model)

    def forward(
            self, query: torch.Tensor,
            key: torch.Tensor = None,
            attn_mask: torch.Tensor = None):
        key = query if key is None else key
        query = query + self.attn(
            query=self.ln_1(query), key=self.ln_2(key), value=self.ln_2(key),
            need_weights=False, key_padding_mask=attn_mask)[0]
        query = query + self.mlp(self.ln_3(query))

        return query


class CrossAttention(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super(CrossAttention, self).__init__()
        self.width = width
        self.layers = layers
        self.blocks = nn.ModuleList([CrossAttentionBLock(width, heads) for _ in range(layers)])
        proj_std = (self.width ** -0.5) * ((2 * self.layers) ** -0.5)
        attn_std = self.width ** -0.5
        fc_std = (2 * self.width) ** -0.5
        for block in self.blocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def forward(self, query: torch.Tensor, key: torch.Tensor, attn_mask: torch.Tensor = None):
        query_size = query.shape
        if len(query.shape) == 4:
            query, key = query.reshape(-1, query.shape[-2], self.width), key.reshape(-1, key.shape[-2], self.width)
        for i in range(self.layers):
            query = self.blocks[i](query=query, key=key, attn_mask=attn_mask)
        return query.view(query_size)


class Alignment(nn.Module):
    """ 参考
    CLIP-Driven Fine-grained Text-Image Person Reidentification
    Unified Coarse-to-Fine Alignment for Video-Text Retrieval
    目的 过滤掉无效信息，然后进行隐式注意力
    """

    def __init__(self, top_k: int, width: int, layers: int, heads: int):
        super(Alignment, self).__init__()
        self.top_k = top_k
        self.crossAttention = CrossAttention(width, layers, heads)
        scale = width ** -0.5
        self.image_class_embedding = nn.Parameter(scale * torch.randn(width))
        self.text_class_embedding = nn.Parameter(scale * torch.randn(width))

    def select_tokens(self, tokens: torch.Tensor, token_scores: torch.Tensor):
        """
        :param tokens: b * b * (m or n)* dim
        :param token_scores: b * b * m * n
        :return: b * b * k * dim 根据相似度选择前k的tokens
        """
        batch_image, batch_text, tokens_len, dim = tokens.shape
        tokens_id = torch.argsort(token_scores, dim=-1, descending=True)[:, :, :self.top_k]
        token_mask = torch.scatter(
            input=torch.zeros((batch_image, batch_text, tokens_len), dtype=torch.int32, device=tokens.device),
            index=tokens_id, value=1, dim=-1)  # b_i, b_t, tokens_len
        token_mask = token_mask.unsqueeze(-1).repeat_interleave(
            repeats=dim, dim=-1)  # b_i, b_t, tokens_len, dim
        tokens = torch.masked_select(input=tokens, mask=token_mask == 1).reshape(
            batch_image, batch_text, self.top_k, -1)  # b_i, b_t, k, dim
        return tokens

    def get_tokens(self, image_tokens: torch.Tensor, text_tokens: torch.Tensor, atte_mask: torch.Tensor):
        batch_image, patch, dim = image_tokens.shape
        batch_text, s, _ = text_tokens.shape
        batch_image, patch, dim = image_tokens.shape
        batch_text, s, _ = text_tokens.shape
        image_tokens = image_tokens / image_tokens.norm(dim=-1, keepdim=True)
        text_tokens = text_tokens / text_tokens.norm(dim=-1, keepdim=True)
        image_tokens = image_tokens.unsqueeze(1).expand(batch_image, batch_text, patch, -1)  # b_i, b_t, patch, dim
        text_tokens = text_tokens.unsqueeze(0).expand(batch_image, batch_text, s, -1)  # b_i, b_t, s, dim

        # calc sim
        sim_i2t = torch.bmm(
            image_tokens.reshape(-1, patch, dim), text_tokens.reshape(-1, s, dim).transpose(1, 2)
        ).reshape(batch_image, batch_text, patch, s)  # b_i * b_text * path * s

        # image top_k
        image_token_scores = sim_i2t.clone().masked_fill(
            atte_mask.unsqueeze(1).unsqueeze(0) == 0, 0).sum(dim=-1, keepdim=False)  # b_i , b_t, patch
        image_selected_tokens = self.select_tokens(image_tokens, image_token_scores)  # b_i, b_t, k, dim

        # text top_k
        text_token_scores = sim_i2t.clone().transpose(2, 3).sum(dim=-1, keepdim=False)  # b_i, b_t, s
        text_selected_tokens = self.select_tokens(text_tokens, text_token_scores)  # b_i, b_t, k, dim

        return image_selected_tokens, text_selected_tokens

    def forward(
            self,
            image_feature: torch.Tensor,
            image_tokens: torch.Tensor,
            text_feature: torch.Tensor,
            text_tokens: torch.Tensor,
            atte_mask: torch.Tensor
    ):
        """
        :param image_feature: b * dim
        :param image_tokens: b * patch * dim
        :param text_feature: b * dim
        :param text_tokens: b * s * dim
        :param atte_mask: b * s
        :return: 细粒度交互后的图像和文本特征
        """
        batch_image, patch, dim = image_tokens.shape
        batch_text, s, _ = text_tokens.shape

        image_feature = image_feature.unsqueeze(1).expand(batch_image, batch_text, 1, dim)
        text_feature = text_feature.unsqueeze(0).expand(batch_image, batch_text, dim).reshape(image_feature.shape)

        image_selected_tokens, text_selected_tokens = self.get_tokens(image_tokens, text_tokens, atte_mask)

        final_image_tokens = self.crossAttention(
            query=torch.cat(
                [self.image_class_embedding.to(image_feature.dtype) + torch.zeros(
                    (batch_image, batch_text, 1, dim), dtype=image_feature.dtype, device=image_feature.device),
                 image_selected_tokens], dim=2),
            key=torch.cat([text_feature, text_selected_tokens], dim=2),
        )  # b_i, b_t, k, dim
        final_text_tokens = self.crossAttention(
            query=torch.cat(
                [self.text_class_embedding.to(text_feature.dtype) + torch.zeros(
                    (batch_image, batch_text, 1, dim), dtype=text_feature.dtype, device=image_feature.device),
                 text_selected_tokens], dim=2),
            key=torch.cat([image_feature, image_selected_tokens], dim=2),
        )  # b_i, b_t, k, dim
        return final_image_tokens, final_text_tokens


if __name__ == "__main__":
    model = Alignment(top_k=4, width=4, layers=2, heads=2)
    output = model.forward(
        image_feature=torch.randn((2, 4)),
        image_tokens=torch.randn((2, 8, 4)),
        text_feature=torch.randn((2, 4)),
        text_tokens=torch.randn((2, 4, 4)),
        atte_mask=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    )
    print(output[0].shape, output[1].shape)
