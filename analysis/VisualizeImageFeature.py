#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/6 16:42
# @Author  : CaoQixuan
# @File    : VisualizeImageFeature.py
# @Description :
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/22 20:41
# @Author  : CaoQixuan
# @File    : test.py
# @Description :
import json

import numpy
import pandas
import seaborn
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn import decomposition
from torch import nn
import os

from torch.utils.data import Dataset
from torchvision.transforms import transforms

from models.base_clip.CLIP import load_clip_from_json


class ImageDataset(Dataset):
    def __init__(self, path: str = None, class_json_path: str = None):
        """
        :param path:
        :param class_json_path: 根据数据集提前生成
        """
        super().__init__()
        self.path = path
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])

        self.file_names = os.listdir(path)
        # 获取当前目录下的所有文件
        self.files_path = [os.path.join(self.path, file) for file in self.file_names]
        self.file2clas = json.load(open(class_json_path, 'r'))

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        img_path = self.files_path[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.file2clas[self.file_names[index]][1]
        return img, label


@torch.no_grad()
def test(model):
    model.eval()
    model.cuda()
    data_loader = torch.utils.data.DataLoader(
        ImageDataset(
            path="./dataset/RSITMD/images/",
            class_json_path="./dataset/RSITMD/image2class.json"
        ),
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    image_features = []
    labels = []
    for data_pair, i in zip(data_loader, (range(len(data_loader)))):
        img, label = data_pair
        img_fea = model(img.cuda()).detach().cpu().numpy()
        image_features.append(img_fea)
        labels.append(label.numpy())
    image_features = numpy.concatenate(image_features, axis=0)
    labels = numpy.concatenate(labels, axis=0)
    return image_features, labels


def get_model(model_name, pretrain):
    self, self.clip_config = load_clip_from_json(
        config_path=r"D:\Code\Pycharm\Project\models_data\base_clip/" + model_name + "/",
        pre_train=True,
    )
    self.eval()
    if pretrain:
        self.load_state_dict(torch.load("D:\Code\Pycharm\Project/analysis/" + model_name + ".pt"))
    return self.visual


if __name__ == '__main__':
    if not os.path.exists("image_features.npy"):
        model = get_model("TinyCLIP-ViT-39M-16-Text-19M-YFCC15M", True)
        image_features, labels = test(model)
        numpy.save("image_features.npy", image_features)
        numpy.save("labels.npy", labels)
    else:
        image_features, labels = numpy.load("image_features.npy"), numpy.load("labels.npy")
    pca = decomposition.PCA(n_components=2)
    pca.fit(image_features)
    features = pca.transform(image_features)
    # data = numpy.concatenate([features, labels], axis=1)

    df = pandas.DataFrame(features[::10], columns=['f1', 'f2'])
    df['label'] = labels[::10]
    plt.figure(figsize=(10, 10), dpi=200)
    seaborn.scatterplot(x='f1', y='f2', hue='label', data=df)
    plt.savefig("image_features.png")
