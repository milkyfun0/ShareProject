import json
from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from collections import OrderedDict
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class Register(nn.Module):

    def __init__(self, num, in_dim, dim, share_dim, dropout=0.1):
        super(Register, self).__init__()

        self.store_info = nn.Parameter(torch.empty(num, dim, dtype=torch.float32))
        self.dim = dim
        self.linear = nn.Linear(in_dim, dim)
        self.base_down = nn.Linear(dim, dim - share_dim)
        self.share_down = nn.Linear(dim, share_dim)
        self.ln_1 = LayerNorm(in_dim)
        self.ln_2 = LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, in_dim),
        )
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.kaiming_normal_(self.linear.weight)
        nn.init.kaiming_normal_(self.base_down.weight)
        nn.init.kaiming_normal_(self.share_down.weight)
        nn.init.kaiming_normal_(self.ffn[0].weight)

    def forward(self, x: torch.Tensor, shared_info: torch.Tensor, attn_mask: torch.Tensor = None):
        res = x
        x = self.linear(x)
        batch = x.shape[0]

        base = self.base_down(self.store_info)
        share = self.share_down(shared_info)
        info = torch.cat([base, share], dim=-1).unsqueeze(dim=0).expand(batch, -1, -1)

        x = F.normalize(x, p=2, dim=-1)
        info = F.normalize(info, p=2, dim=-1)

        scores = torch.bmm(x, info.transpose(1, 2))
        if attn_mask is not None:
            scores.masked_fill_(attn_mask == 1, -float('inf'))
        attn = torch.softmax(scores, dim=-1)

        out = torch.bmm(attn, info)
        out = self.ffn(self.ln_2(out)) + res

        return torch.relu(out)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
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

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, key_padding_mask=attn_mask)[
            0]  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, register: Register = None,
                share_info: torch.Tensor = None):
        for i in range(self.layers):
            if i == self.layers // 2 and register is not None:
                x = register(x, share_info)
            x = self.resblocks[i](x, attn_mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 opt):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # width相当于transform中的d_model
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        self.register = Register(num=opt["model"]["register"]["num"], in_dim=width,
                                 dim=width, share_dim=opt["model"]["register"]["share_dim"])
        self.image_info = self.register.store_info

    def forward(self, x: torch.Tensor, share_info: torch.Tensor = None):
        # x=[1,3,224,224]
        x = self.conv1(x)  # shape = [*, width, grid, grid] # 将图片分成[32,32]个patch [1,768,7,7]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2],合并高宽 [1,768,49]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] ，更换位置 [1,49,768]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width],添加cls token[1,50,768]
        x = x + self.positional_embedding.to(x.dtype)  # 这里位置编码是可学习的参数，可能是切了path顺序让模型自己学习吧  [1,50,768]
        x = self.ln_pre(x)  # [1,50,768]

        x = x.permute(1, 0, 2)  # NLD -> LND  # [pixel,b,d_model]=[50,1,768]
        # 当实例化时 batch——first默认维false
        x = self.transformer(x, register=self.register, share_info=share_info)  # 多头transformer [50,1,768]
        x = x.permute(1, 0, 2)  # LND -> NLD  # [1,50,768]

        x = self.ln_post(x[:, 0, :])  # x[:, 0, :] 将所有信息汇聚到cls token中，只需前面来做下游任务 [1,768]

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
                 opt
                 ):
        super().__init__()

        self.context_length = context_length  # 77
        self.Eiters = 0

        vision_heads = vision_width // 64
        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
            opt=opt
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

        self.register = Register(num=opt["model"]["register"]["num"], in_dim=transformer_width,
                                 dim=transformer_width, share_dim=opt["model"]["register"]["share_dim"])
        self.text_info = self.register.store_info

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
        return self.visual(image.type(self.dtype), share_info=self.text_info)

    def encode_text(self, text, attn_mask: Optional[torch.Tensor] = None):
        # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
        x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
        x = self.transformer(x, attn_mask, register=self.register,
                             share_info=self.visual.image_info)  # 共11个 和图像encode结构一致 [77,3,512]
        x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # text.argmax(dim=-1) 句子最后有一个seq字段，是最大的，因此能获得句子个数数量
        x = x[:, -1, :] @ self.text_projection

        return x

    def forward(self, image: torch.Tensor, text, attn_mask):
        """
        :param image:
        :param text: (text_id, attn_mask)
        :return:
        """
        attn_mask = (attn_mask == 0)
        # if type(text) is Tuple:
        # attn_mask = None
        #     text, _, attn_mask = text
        #     print(len(text))
        # else:
        #     attn_mask = None
        batch_image, batch_text = len(image), len(text)
        image_features = self.encode_image(image)
        text_features = self.encode_text(text, attn_mask)

        # normalized features,# 每一行sqr(a1^2+a2^2+...)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)  # [batch_img,512]
        # text_features = text_features / text_features.norm(dim=1, keepdim=True)  # [batch_text,512]

        # # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()  # 可学习参数
        # logits_per_image = logit_scale * image_features @ text_features.t()  # 特征相乘获得相似度
        # logits_per_text = logits_per_image.t()  # 变成文本

        image_features = image_features.unsqueeze(dim=1).expand(-1, batch_text, -1)
        text_features = text_features.unsqueeze(dim=0).expand(batch_image, -1, -1)

        dual_sim = cosine_similarity(image_features, text_features)

        return dual_sim


def get_freeze_layers_names():
    names = [
        "positional_embedding",
        "visual.class_embedding",
        "visual.positional_embedding",
        "visual.conv1.weight",
        "visual.conv1.bias",
    ]
    for i in range(5):
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

    for i in range(2):
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
    return new_dict


def load_clip_from_json(config_path: str, pre_train: bool = True, opt=None):
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
        opt=opt
    )
    if pre_train:
        weights_map = trans_standard_static_dict(torch.load(config_path + "/" + model_cfg["name"]))
        model.load_state_dict(weights_map, strict=False)

        freeze_names = get_freeze_layers_names()
        for name, param in model.named_parameters():
            if name in freeze_names:
                param.requires_grad = False
    return model, model_cfg
