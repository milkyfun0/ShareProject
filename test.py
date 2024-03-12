import random

import torch
from mpmath import li
from torch import nn
from transformers import AutoTokenizer


def trans_token_embedding(static_dict_path, store_path, tokenizer_path, vocab_path, special_token_trans):
    static_dict = torch.load(static_dict_path)
    n, d = static_dict["token_embedding.weight"].shape
    token_embedding = nn.Embedding(n, d).from_pretrained(
        static_dict["token_embedding.weight"])
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    with open(vocab_path) as f:
        vocab = f.readlines()

    def function(word):
        word = word.replace("\n", "")
        if word in special_token_trans:
            word = special_token_trans[word]
        input_ids = tokenizer([word], return_tensors='pt', padding=False)["input_ids"]
        print(input_ids)
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
    return new_tokens_id


if __name__ == '__main__':
    trans_dict = {
        "[PAD]": "<|endoftext|>",
        "[UNK]": "<|endoftext|>",
        "[CLS]": "<|startoftext|>",
        "[SEP]": "<|endoftext|>",
        "[MASK]": "<|endoftext|>"
    }
    # trans_token_embedding(
    #     static_dict_path="./models_data/base_clip/TinyCLIP-ViT-40M-32-Text-19M-LAION400M/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt",
    #     tokenizer_path="models_data/base_clip/tokenizer",
    #     vocab_path="temp/vocab.txt",
    #     special_token_trans=trans_dict
    # )  # 投影转换
