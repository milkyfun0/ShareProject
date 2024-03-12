"""
@File  :Moudules.py
@Author:CaoQiXuan
@Date  :24/2/2414:59
@Desc  :
"""
import math
from collections import OrderedDict

import torch
from torch import nn
from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Register(nn.Module):

    def __init__(self, num, in_dim, dim, share_dim, dropout=0.1):
        super(Register, self).__init__()

        self.store_info = nn.Parameter(torch.empty(num, dim, dtype=torch.float32))
        self.dim = dim
        self.linear = nn.Linear(in_dim, dim)
        self.base_down = nn.Linear(dim, dim - share_dim)
        self.share_down = nn.Linear(dim, share_dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, in_dim),
            # nn.Dropout(dropout),
            # nn.Tanh(),
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
        info = torch.cat([base, share], dim=1).unsqueeze(dim=0).expand(batch, -1, -1)
        scores = torch.bmm(x, info.transpose(1, 2)) / math.sqrt(self.dim)
        # if attn_mask is not None:
        #     scores.masked_fill_(attn_mask == 1, -float('inf'))
        attn = torch.softmax(scores, dim=-1)

        out = torch.bmm(attn, info)
        out = self.ffn(out) + res

        return torch.relu(out)


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

    def attention(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask == 1
        return self.attn(x, x, x, need_weights=False, key_padding_mask=attn_mask)[
            0]  # 三个x表示Q K V计算值，x最后维度=n_head*d_model

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for i in range(self.layers):
            x = self.resblocks[i](x, attn_mask)
        return x


class TextEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.embed_dim = opt["model"]["dim"]
        self.context_length = opt["model"]["context_length"]
        self.vocab_size = opt["model"]["text"]["vocab_size"]
        self.width = opt["model"]["text"]["width"]
        self.heads = opt["model"]["text"]["heads"]
        self.layers = opt["model"]["text"]["layers"]

        self.transformer = Transformer(
            width=self.width,
            layers=self.layers,
            heads=self.heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = nn.Embedding(self.vocab_size, self.width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, self.width))
        self.ln_final = LayerNorm(self.width)
        self.text_projection = nn.Parameter(torch.empty(self.width, self.embed_dim))

        self.register = Register(num=opt["model"]["register"]["num"], in_dim=self.width,
                                 dim=opt["model"]["register"]["dim"], share_dim=opt["model"]["register"]["share_dim"])
        self.text_info = self.register.store_info
        # self.initialize_parameters()

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

    @property
    def dtype(self):
        return torch.float32

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, text, share_info: torch.Tensor, attn_mask: torch.Tensor = None):
        # x 每个句子前面有值，有2个特殊符号[CLS]与[Seq]
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]，[3,77,512]
        x = x + self.positional_embedding.type(self.dtype)  # 位置编码直接赋可学习位置，添加位置信息[3,77,512]
        x = x.permute(1, 0, 2)  # NLD -> LND,[77,3,512]
        x = self.transformer(x, attn_mask == 0)  # 共11个 和图像encode结构一致 [77,3,512]
        x = x.permute(1, 0, 2)  # LND -> NLD，[3,77,512]
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # text.argmax(dim=-1) 句子最后有一个seq字段，是最大的，因此能获得句子个数数量

        cls_token = x[:, 0, :]
        # tokens = self.register(x, share_info)
        # sum_tokens = torch.sum(tokens * (torch.ones_like(tokens) - attn_mask.unsqueeze(-1)), dim=-2, keepdim=False)
        # avg_tokens = sum_tokens / torch.sum(torch.ones_like(attn_mask) - attn_mask, dim=-1, keepdim=True)
        # # avg_tokens = tokens.mean(dim=1, keepdim=False)
        # out = (cls_token + avg_tokens) @ self.text_projection
        out = cls_token @ self.text_projection
        return out


class VisualEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # if opt["model"]["visual"]["name"] is "resnet18":
        # self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        # exec("self.resnet = " + opt["model"]["visual"]["name"] + "(pretrained=True)")

        self.fc = nn.Linear(self.resnet.fc.in_features, opt['model']["dim"])
        self.register = Register(num=opt["model"]["register"]["num"], in_dim=opt["model"]["dim"],
                                 dim=opt["model"]["register"]["dim"], share_dim=opt["model"]["register"]["share_dim"])
        self.image_info = self.register.store_info

    def forward(self, images: torch.Tensor, share_info: torch.Tensor):
        x = self.resnet.conv1(images)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        f1 = self.resnet.layer1(x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)

        # use self attention
        f4 = self.resnet.layer4(f3)
        b, c, h, w = f4.size()
        # f4 = self.register(f4.view(b, c, -1).transpose(1, 2), share_info)
        # features = torch.mean(f4, dim=1, keepdim=False)
        # features = self.fc(features)
        # print(self.fc.weight.grad)

        features = self.resnet.avgpool(f4)
        features = torch.flatten(features, 1)
        # linear projection to the joint embedding space
        features = self.fc(features)

        return features
