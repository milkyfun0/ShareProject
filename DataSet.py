"""
@File  :DataSet.py
@Author:CaoQiXuan
@Date  :23/12/314:38
@Desc  :
"""
import json
import os
import random

import numpy
import torch
import torch.utils.data as data
from PIL import Image
from torch import nn
from torchvision import transforms
from transformers import AutoTokenizer, CLIPTokenizerFast


def trans_token_embedding(static_dict_path, store_path, tokenizer, vocab_path, special_token_trans):
    static_dict = torch.load(static_dict_path)
    if "state_dict" in static_dict:
        static_dict = static_dict["state_dict"]
    token_embedding = None
    for k, v in static_dict.items():
        if "token_embedding.weight" in k:
            token_embedding = v
    n, d = token_embedding.shape
    token_embedding = nn.Embedding(n, d).from_pretrained(
        token_embedding)

    vocab = ["<|startoftext|>", "<|endoftext|>"]
    with open(vocab_path) as f:
        vocab += f.readlines()

    def function(word):
        word = word.replace("\n", "")
        if word in special_token_trans:
            word = special_token_trans[word]
        input_ids = tokenizer([word], return_tensors='pt', padding=False)["input_ids"]
        token_id = input_ids[:, 1:-1].squeeze(0)
        token = token_embedding(input_ids)[:, 1:-1].squeeze(0)
        return token, token_id

    new_token_embedding, new_tokens_id = [], []
    for token, token_id in map(function, vocab):
        new_token_embedding.append(token)
        new_tokens_id.append(token_id)

    new_token_embedding = torch.cat(new_token_embedding, dim=0)
    new_tokens_id = torch.cat(new_tokens_id, dim=0)
    torch.save(new_token_embedding, store_path + "token_embedding.tensor")
    return new_tokens_id.tolist()


class RSTIMD_DataSet(data.Dataset):
    def __init__(self, opt, path: str, tokenizer_path: str, context_length: int, img_div, flag: str, **kwargs):
        super(RSTIMD_DataSet).__init__()
        assert flag in ("train", "test", "val")
        self.path = path
        self.context_length = context_length
        self.img_div = img_div
        self.captions = []
        self.images_name = []
        with open(path + '%s_caps.txt' % flag, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                self.captions.append(self.process(line))
        with open(path + '%s_filename.txt' % flag, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                self.images_name.append(line.strip())

        self.image2class = get_image_class_from_filename(opt)
        self.length = len(self.captions)

        if flag == "train":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.RandomRotation(0, InterpolationMode.BILINEAR),
                # transforms.RandomCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        self.MLM = opt["model"]["MLM"]
        self.isTrans = opt["dataset"]["isTrans"]
        # if opt["model"]["name"] in ["base_clip", "base_register", "base_peg"]:
        self.tokenizer = CLIPTokenizerFast.from_pretrained(tokenizer_path)
        # CLIPTokenizerFast
        # else:
        #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        # BertTokenizer
        if self.isTrans:
            self.tokens_ids = trans_token_embedding(
                static_dict_path=opt["model"]["CLIP"]["config_path"] + "/" + opt["model"]["CLIP"]["save_name"],
                store_path="models_data/" + opt["model"]["name"] + "/",
                tokenizer=self.tokenizer,
                vocab_path="models_data/" + opt["model"]["name"] + "/vocab.txt",
                special_token_trans={
                    "[PAD]": "<|endoftext|>",
                    "[UNK]": "<|endoftext|>",
                    "[CLS]": "<|startoftext|>",
                    "[SEP]": "<|endoftext|>",
                    "[MASK]": "<|endoftext|>"
                }
            )

    @staticmethod
    def process(s: str):
        for i in [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']:
            s = s.strip().lower().replace(i, " " + i + " ")
        return s.strip()

    def __getitem__(self, index):
        img_id = index // self.img_div
        caption = self.tokenizer(
            [self.captions[index]], return_tensors='pt', padding="max_length",
            max_length=self.context_length, truncation=True)  # {input_ids, attention_mask}

        input_ids, attention_mask = caption['input_ids'].squeeze(), caption['attention_mask'].squeeze()

        input_ids = input_ids.apply_(self.trans_id)

        if self.MLM:
            mask_caption, mask_label = self.build_masked_tokens(caption["input_ids"].squeeze())
        else:
            mask_caption, mask_label = None, None

        image = Image.open(self.path + "images/" + str(self.images_name[img_id])).convert('RGB')
        image = self.transform(image)

        return {
            "images_id": int(img_id),
            "images": image,
            "images_class": self.image2class[str(self.images_name[img_id])][1],
            "text_id": int(index),
            "text": input_ids,
            "attn_mask": attention_mask,
            "mask_caption": mask_caption,
            "mask_label": mask_label
        }

    def trans_id(self, value):
        if not self.isTrans:
            return value
        if self.tokens_ids.count(value) == 0:
            return 1
        return self.tokens_ids.index(value)

    def build_masked_tokens(self, tokens):
        mask = self.trans_id(self.tokenizer(["<|endoftext|>"])["input_ids"][0][1])
        if self.isTrans:
            low, high = 2, len(self.tokens_ids)
        else:
            low, high = 0, self.tokenizer.vocab_size - 2

        mask_id = (torch.rand(tokens.shape) < 0.15) & (low <= tokens) & (tokens < high)
        prob = torch.rand(tokens.shape)
        labels = mask_id * tokens + mask_id * 1 + -1  # -1 is not a ward
        # [0, 0.15*0.8]:mask  [0.15*0.8, 0.15*0.9]:random  [0.15*0.9, 0.15]:not change
        tokens = (~mask_id + (mask_id & (0.9 <= prob))) * tokens + (mask_id & (prob < 0.8)) * mask + (
                mask_id & (0.8 <= prob) & (prob < 0.9)) * torch.randint(low, high, tokens.shape)
        return tokens, labels

    def __len__(self):
        return self.length


def get_image_class_from_filename(opt):
    """
    just support  RSTIMD and RSICD
    :param opt:
    :return:
    """
    # if os.path.exists(opt["dataset"]["path"] + "image2class.json"):
    #     with open(opt["dataset"]["path"] + "image2class.json", 'r') as f:
    #         return json.load(f)

    with open(opt["dataset"]["path"] + "dataset_RSITMD.json", 'r') as f:
        dataset_info = json.load(f)
    image2class = {}
    class_name = {}
    class_count = 0
    for image in dataset_info["images"]:
        file_name = image["filename"]
        temp = file_name.split("_")[0]
        if temp not in class_name:
            class_name[temp] = class_count
            class_count += 1
        image2class[file_name] = file_name.split("_")[0], class_name[temp]
    opt["dataset"]["images_class"] = class_count
    with open(opt["dataset"]["path"] + "image2class.json", "w", encoding="utf-8") as file:
        json.dump(image2class, file)
        return image2class


def get_loader(opt):
    if opt["dataset"]["name"] == "RSITMD":
        train_loader = torch.utils.data.DataLoader(
            dataset=RSTIMD_DataSet(
                opt=opt,
                path=opt["dataset"]["path"],
                tokenizer_path="models_data/" + opt["model"]["name"] + "/tokenizer",
                context_length=opt["model"]["context_length"],
                img_div=opt["dataset"]["img_div"],
                flag="train"
            ),
            batch_size=opt["train"]["batch_size"],
            num_workers=opt["train"]["num_works"],
            shuffle=True,
            pin_memory=True,
            drop_last=False
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=RSTIMD_DataSet(
                opt=opt,
                path=opt["dataset"]["path"],
                tokenizer_path="models_data/" + opt["model"]["name"] + "/tokenizer",
                context_length=opt["model"]["context_length"],
                img_div=opt["dataset"]["img_div"],
                flag="val"
            ),
            batch_size=opt["train"]["batch_size_test"],
            num_workers=opt["train"]["num_works"],
            shuffle=False,
            pin_memory=True
        )
        return train_loader, test_loader
    else:
        assert False  # not such dataset


if __name__ == "__main__":
    from main import get_options

    train_loader, test_loader = get_loader(get_options())
    for data_map in train_loader:
        print(data_map["text"][0])
        print(data_map["attn_mask"][0])
        print(data_map["mask_caption"][0])
        print(data_map["mask_label"][0])
        break
#     "text_id": int(index),
#             "text": caption["input_ids"].squeeze(),
#             "attn_mask": caption["attention_mask"].squeeze(),
#             "mask_caption": mask_caption,
#             "mask_label": mask_label
# 49406,   530,   518,  4331, 49407,   518,  1901, 10750, 49407, 49407
