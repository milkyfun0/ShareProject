#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/4 17:01
# @Author  : CaoQixuan
# @File    : main.py
# @Description :

from Collection import VarCollection, visualize_grid_to_grid_with_cls, visualize_grid_to_grid, visualize_head, \
    visualize_heads

VarCollection.activate()
from PIL import Image
import seaborn
import torch
from torch import nn
import numpy
from models.base_clip.CLIP import load_clip_from_json
import torch
from torch import nn
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))])


def get_attention_map(model_name, pretrain, image):
    self, self.clip_config = load_clip_from_json(
        config_path=r"D:\Code\Pycharm\Project\models_data\base_clip/" + model_name + "/",
        pre_train=True,
    )
    self.eval()
    if pretrain:
        self.load_state_dict(torch.load("D:\Code\Pycharm\Project/analysis/data/" + model_name + ".pt"))
    visual_encoder = self.visual
    VarCollection.clear()
    with torch.no_grad():
        feature = visual_encoder(image_tensor)
    cache = VarCollection.cache
    return cache['ResidualAttentionBlock.attention']


# TinyCLIP-ViT-61M-32-Text-29M-LAION400M
# TinyCLIP-ViT-39M-16-Text-19M-YFCC15M
# TinyCLIP-ViT-40M-32-Text-19M-LAION400M
# D:\Code\Pycharm\Project/analysis/
# D:\Code\Pycharm\Project\dataset\RSITMD\images/

if __name__ == '__main__':
    image_path = r"./img/airport_7.tif"
    model_name = "TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
    image = Image.open(image_path).convert('RGB')
    image_tensor = image_transform(image).unsqueeze(0)
    image = image.resize((224, 224))
    attention_maps = get_attention_map(
        model_name=model_name,
        pretrain=True,
        image=image_tensor
    )
    print(attention_maps[0].shape)
    layers = len(attention_maps)
    heads = attention_maps[0].shape[1]
    layer = 5
    for i in range(heads):
        visualize_grid_to_grid_with_cls(attention_maps[layer][0, i, :, :], 0, image, grid_size=14,
                                        title="layer={} head={} ".format(layer, i) + model_name)
        # break
    for i in range(heads):
        visualize_grid_to_grid(attention_maps[layer][0, i, 1:, 1:], 77, image, grid_size=14,
                               title="layer={} head={} ".format(layer, i) + model_name)  # 22
    # # #
    # visualize_heads(attention_maps[1], cols=3, title="layer={}".format(layer))
