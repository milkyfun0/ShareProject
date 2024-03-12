"""
@File  :utils.py
@Author:CaoQiXuan
@Date  :23/12/3 16:52
@Desc  :
 """
import argparse
import json
import os
import random
import time
from copy import copy
from typing import Optional, Union

import numpy
import torch
import yaml
from torch import nn
from tqdm import tqdm


def same_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    numpy.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def params_count(model):
    count = 0
    for p in model.parameters():
        c = 1
        for i in range(p.dim()):
            c *= p.size(i)
        count += c
    return count


def get_options(model_name="base_clip"):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_opt', default="models_data/" + model_name + "/config.yaml", type=str,
                        help='path to a yaml options file')
    opt = parser.parse_args()
    with open(opt.path_opt, 'r') as handle:
        options = yaml.load(handle, Loader=yaml.FullLoader)

    return options


def cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, dim=-1, eps=1e-4):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    if len(x1.shape) == 2:  # tensor is 2D
        x1 = x1.unsqueeze(dim=1).expand(-1, x2.shape[0], -1)
        x2 = x2.unsqueeze(dim=0).expand(x1.shape[0], x2.shape[0], -1)

    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def get_retrieved_images_index(sims, k=5, percent=1):
    top_k = numpy.argsort(-sims, axis=1)[:, :k].reshape(-1)
    retrieved_images_index, counts = numpy.unique(top_k, return_counts=True)
    return retrieved_images_index[:int(len(retrieved_images_index) * percent)]


def get_index_to_normalize(sims, images_index):
    arg_m = numpy.argsort(-sims, axis=1)[:, 0]
    result = numpy.array(list(map(lambda x: x in images_index, arg_m)))
    result = numpy.nonzero(result)
    return result  # 返回需要归QBNorm的下标


def QB_Norm(sim: numpy.ndarray, sample_percent=0.2, top_k=5, top_percent=1, beta=20):
    """
    仅用于文本检索图片
    https://arxiv.org/abs/2112.12777
    :param sim: (text_len, image_len)
    :param sample_percent:
    :param top_k:
    :param top_percent:
    :param beta:
    :return:
    """

    train_index = numpy.random.choice(len(sim), int(len(sim) * sample_percent), replace=False)
    train_test = numpy.exp(sim[train_index] * beta)
    test_test = numpy.exp(sim * beta)

    retrieved_images_index = get_retrieved_images_index(train_test, k=top_k, percent=top_percent)
    normalizing_sum = numpy.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_images_index)
    sim[index_for_normalizing, :] = \
        numpy.divide(test_test[index_for_normalizing, :], normalizing_sum).clip(min=1e-8)
    return sim


def metricR(ranks: Union[numpy.ndarray, torch.Tensor]):
    """
    :param ranks: (-1, 1)
    :return:
    """
    total = len(ranks)
    if isinstance(ranks, numpy.ndarray):
        r1 = (ranks == 0).astype(int).sum() * 100 / total
        r5 = (ranks < 5).astype(int).sum() * 100 / total
        r10 = (ranks < 10).astype(int).sum() * 100 / total
    else:
        r1 = (ranks == 0).to(int).sum() * 100 / total
        r5 = (ranks < 5).to(int).sum() * 100 / total
        r10 = (ranks < 10).to(int).sum() * 100 / total
    return r1, r5, r10, (r1 + r5 + r10) / 3


def acc_i2t_numpy(
        sim: numpy.ndarray,
        img_div: int,
        image_label: Optional[numpy.ndarray] = None,
        text_label: Optional[numpy.ndarray] = None,
):
    """
    20231205
    :param sim: (image_len, text_len)
    :param img_div:
    :param image_label: (image_len,)
    :param text_label: (text_len,)
    :return: R1, R5, R10, mR, ranks
    input examples:
    1. sim: (100,400) img_div: 4 ok
    2. sim: (100,400) img_div: 4 img_label: (100,1) text_label: (400,1) ok
    3. sim: (400,400) img_div: 4 ok
    4. sim: (400,400) img_div: 1 img_label: (400,1) text_label: (400,1)
    5. sim: (400,400) img_div: 4 img_label: (400,1) text_label: (400,1) ok
    """
    assert not ((image_label is None) ^ (text_label is None))
    image_len, text_len = sim.shape
    scale = text_len // image_len

    if image_label is None:
        image_label = numpy.array(range(image_len)).reshape(-1, 1)
        image_label = image_label if scale == img_div else image_label // img_div
        text_label = numpy.array(range(text_len)).reshape(1, -1)

    image_label = numpy.repeat(image_label.reshape(-1, 1), text_len, axis=1)
    text_label = numpy.repeat(text_label.reshape(1, -1), image_len, axis=0) // img_div

    pre_ids = text_label[numpy.array(range(image_len)).reshape(-1, 1), numpy.argsort(sim, axis=1)][:, ::-1]
    # pre_ids = text_label[numpy.array(range(image_len)).reshape(-1, 1), numpy.argsort(-sim, axis=1)]
    # is not same with up  because is sort of sable

    ranks = numpy.where(pre_ids == image_label)[1].reshape(image_len, -1)[:, 0]
    return metricR(ranks), ranks


def acc_t2i_numpy(
        sim: numpy.ndarray,
        img_div: int,
        image_label: Optional[numpy.ndarray] = None,
        text_label: Optional[numpy.ndarray] = None,
):
    """
    20231205
    :param sim: (image_len, text_len)
    :param img_div:
    :param image_label: (image_len,)
    :param text_label: (text_len,)
    :return: R1, R5, R10, mR, ranks
    input examples:
    1. sim: (400,100) img_div: 4 ok
    2. sim: (400,100) img_div: 4 img_label: (100,1) text_label: (400,1) ok
    3. sim: (400,400) img_div: 4 ok
    4. sim: (400,400) img_div: 1 img_label: (400,1) text_label: (400,1)
    5. sim: (400,400) img_div: 4 img_label: (400,1) text_label: (400,1) ok
    test code :
        mian()
            arr = numpy.load("distance_image2text.npy")
            # test 400 * 1000 result: ok
            data_pair = acc_i2t_numpy(arr, img_div=5)
            print(data_pair[0])
            data_pair = acc_t2i_numpy(arr.T, img_div=5)
            print(data_pair[0])
            # test 4000 * 4000 result: ok test
            index = numpy.repeat(numpy.array(range(800)), 5)
            sim = numpy.zeros((4000, 4000))
            sim[:] = arr[index]
            img_label = numpy.array(range(4000)) // 5
            text_label = numpy.array(range(4000))
            data_pair = acc_i2t_numpy(sim, img_div=5, image_label=img_label, text_label=text_label)
            print(data_pair[0])
            data_pair = acc_t2i_numpy(sim.T, img_div=5, image_label=img_label, text_label=text_label)
            print(data_pair[0])
            print(arr.shape)
    """
    assert not ((image_label is None) ^ (text_label is None))
    text_len, image_len = sim.shape
    scale = text_len // image_len
    if image_label is None:
        image_label = numpy.array(range(image_len)).reshape(1, -1)
        image_label = image_label if scale == img_div else image_label // img_div
        text_label = numpy.array(range(text_len)).reshape(-1, 1)

    image_label = numpy.repeat(image_label.reshape(1, -1), text_len, axis=0)
    text_label = numpy.repeat(text_label.reshape(-1, 1), image_len, axis=1) // img_div

    pre_ids = image_label[numpy.array(range(text_len)).reshape(-1, 1), numpy.argsort(sim, axis=1)][:, ::-1]
    # pre_ids = text_label[numpy.array(range(image_len)).reshape(-1, 1), numpy.argsort(-sim, axis=1)]
    # is not same with up  because is sort of sable
    ranks = numpy.where(pre_ids == text_label)[1].reshape(text_len, -1)[:, 0] * scale // img_div
    return metricR(ranks), ranks


@torch.no_grad()
def acc_i2t_tensor(
        sim: torch.Tensor,
        img_div: int,
        image_label: torch.Tensor = None,
        text_label: torch.Tensor = None
):
    """
    20231206 same to acc_i2t_numpy
    :param sim:
    :param img_div:
    :param image_label:
    :param text_label:
    :return:
    test code:
        # arr = numpy.load("distance_image2text.npy")
        # # arr = torch.tensor(arr)
        # # test 400 * 1000 result: ok
        # # data_pair = acc_i2t_tensor(arr, img_div=5)
        # # print(data_pair[0])
        # # data_pair = acc_t2i_tensor(arr.T, img_div=5)
        # # print(data_pair[0])
        # # # test 4000 * 4000 result: ok test
        # index = numpy.repeat(numpy.array(range(800)), 5)
        # sim = numpy.zeros((4000, 4000))
        # sim[:] = arr[index]
        # sim = torch.tensor(sim)
        # img_label = torch.tensor(numpy.array(range(4000)) // 5)
        # text_label = torch.tensor(numpy.array(range(4000)))
        # data_pair = acc_i2t_tensor(sim, img_div=5, image_label=None, text_label=None)
        # print(data_pair[0])
        # data_pair = acc_t2i_tensor(sim.T, img_div=5, image_label=None, text_label=None)
        # print(data_pair[0])
        # # print(arr.shape)
    """
    assert not ((image_label is None) ^ (text_label is None))
    image_len, text_len = sim.shape
    scale = text_len // image_len
    if image_label is None:
        image_label = torch.tensor(range(image_len), device=sim.device)
        image_label = image_label if scale == img_div else torch.div(image_label, img_div, rounding_mode="floor")
        text_label = torch.tensor(range(text_len), device=sim.device)
    image_label = image_label.reshape(-1, 1).expand(image_len, text_len)
    text_label = torch.div(text_label.reshape(1, -1).expand(image_len, text_len), img_div, rounding_mode="floor")

    pre_ids = torch.gather(text_label, dim=1, index=torch.argsort(sim, dim=1, descending=True))
    ranks = torch.where(pre_ids == image_label)[1].reshape(image_len, -1)[:, 0]
    return metricR(ranks), ranks


@torch.no_grad()
def acc_t2i_tensor(
        sim: torch.Tensor,
        img_div: int,
        image_label: torch.Tensor = None,
        text_label: torch.Tensor = None
):
    """
    20231206 same to acc_t2i_numpy
    :param sim:
    :param img_div:
    :param image_label:
    :param text_label:
    :return:
    """
    assert not ((image_label is None) ^ (text_label is None))
    text_len, image_len = sim.shape
    scale = text_len // image_len
    if image_label is None:
        image_label = torch.tensor(range(image_len), device=sim.device)
        image_label = image_label if scale == img_div else torch.div(image_label, img_div, rounding_mode="floor")
        text_label = torch.tensor(range(text_len), device=sim.device)
    image_label = image_label.reshape(1, -1).expand(text_len, image_len)
    text_label = torch.div(text_label.reshape(-1, 1).expand(text_len, image_len), img_div, rounding_mode="floor")

    pre_ids = torch.gather(image_label, dim=1, index=torch.argsort(sim, dim=1, descending=True))
    ranks = torch.div(
        torch.where(pre_ids == text_label)[1].reshape(text_len, -1)[:, 0] * scale, img_div, rounding_mode="floor")
    return metricR(ranks), ranks


def validate_on_numpy(opt, sim: torch.Tensor, image_labels: torch.Tensor = None, text_labels: torch.Tensor = None):
    sim, image_labels, text_labels = sim.cpu().numpy(), image_labels.cpu().numpy(), text_labels.cpu().numpy()

    i2t_r, i2t_ranks = acc_i2t_numpy(sim=sim, img_div=opt["dataset"]["img_div"], image_label=image_labels,
                                     text_label=text_labels)
    t2i_r, t2i_ranks = acc_t2i_numpy(sim=sim.T, img_div=opt["dataset"]["img_div"], image_label=image_labels,
                                     text_label=text_labels)
    return {
        "i2t": {
            "R1": i2t_r[0],
            "R5": i2t_r[1],
            "R10": i2t_r[2],
            "mR": i2t_r[3]
        },
        "t2i": {
            "R1": t2i_r[0],
            "R5": t2i_r[1],
            "R10": t2i_r[2],
            "mR": t2i_r[3]
        },
        "mR": (i2t_r[3] + t2i_r[3]) / 2
    }


@torch.no_grad()
def validate_on_tensor(opt, sim: torch.Tensor, image_labels: torch.Tensor = None, text_labels: torch.Tensor = None):
    i2t_r, i2t_ranks = acc_i2t_tensor(sim=sim, img_div=opt["dataset"]["img_div"], image_label=image_labels,
                                      text_label=text_labels)
    t2i_r, t2i_ranks = acc_t2i_tensor(sim=sim.T, img_div=opt["dataset"]["img_div"], image_label=image_labels,
                                      text_label=text_labels)
    return {
        "i2t": {
            "R1": i2t_r[0].cpu().item(),
            "R5": i2t_r[1].cpu().item(),
            "R10": i2t_r[2].cpu().item(),
            "mR": i2t_r[3].cpu().item()
        },
        "t2i": {
            "R1": t2i_r[0].cpu().item(),
            "R5": t2i_r[1].cpu().item(),
            "R10": t2i_r[2].cpu().item(),
            "mR": t2i_r[3].cpu().item()
        },
        "mR": (i2t_r[3].cpu().item() + t2i_r[3].cpu().item()) / 2,
        "sum": i2t_r[0].cpu().item() + i2t_r[1].cpu().item() + i2t_r[2].cpu().item() + t2i_r[0].cpu().item() + t2i_r[
            1].cpu().item() + t2i_r[2].cpu().item()
    }


@torch.no_grad()
def validate_with_no_cross(test_loader, model, opt):
    """
    this is a non-cross modal validate way, free more time
    :param test_loader:
    :param model:
    :param opt:
    :return:
    test code :
        main()
            opt = get_options()
            train_loader, test_loder = get_loader(opt)
            model = Network(opt=opt).cuda()
            print(validate_with_no_cross(test_loader=test_loder, model=model, opt=opt))
        validate_with_no_cross():
            numpy.save("./test.npy", sim)
            numpy.save("./image_label.npy", image_labels)
            numpy.save("./text_label.npy", text_labels)
    """
    model_device = next(model.parameters()).device
    test_len = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    data_type = torch.float32

    image_features = torch.empty((test_len, opt["model"]["dim"]), device=model_device, dtype=data_type)
    text_features = torch.empty((test_len, opt["model"]["dim"]), device=model_device, dtype=data_type)
    image_labels = torch.empty(test_len, device=model_device, dtype=data_type)
    text_labels = torch.empty(test_len, device=model_device, dtype=data_type)
    sim = torch.empty(test_len, test_len, device=model_device, dtype=data_type)

    for data_pair, i in zip(test_loader, tqdm(range(len(test_loader)))):
        images_id, images, text_id, text, attn_mask = data_pair["images_id"].to(model_device), data_pair["images"].to(
            model_device), data_pair[
            "text_id"].to(model_device), data_pair["text"].to(model_device), data_pair["attn_mask"].to(model_device)
        batch_len = len(images)
        image_feature = model.encode_image(images)
        text_feature = model.encode_text(text=text, attn_mask=attn_mask)
        image_features[batch_size * i:batch_size * i + batch_len] = image_feature
        text_features[batch_size * i:batch_size * i + batch_len] = text_feature
        image_labels[batch_size * i:batch_size * i + batch_len] = images_id
        text_labels[batch_size * i:batch_size * i + batch_len] = text_id

        torch.cuda.empty_cache()  # free GPU

    for row in range((len(test_loader))):  # time change GPU mem
        for col in range(len(test_loader)):
            sim[batch_size * row:batch_size * (row + 1), batch_size * col:batch_size * (col + 1)] = cosine_similarity(
                image_features[batch_size * row:batch_size * (row + 1)],
                text_features[batch_size * col:batch_size * (col + 1)])

    return validate_on_tensor(opt, sim, image_labels, text_labels)


@torch.no_grad()
def validate_with_cross(test_loader, model, opt):
    """
    this function need to be tested 20231207
    same function as validate_with_no_cross(), this is a cross modal validate way, cost more time
    :param test_loader:
    :param model:
    :param opt:
    :return:
    """
    model_device = next(model.parameters()).device
    test_len = len(test_loader.dataset)
    batch_size = test_loader.batch_size
    data_type = torch.float32
    test_loader_iter = copy(test_loader)

    sim = torch.empty((test_len, test_len), device=model_device, dtype=data_type)
    image_labels = torch.empty(test_len, device=model_device, dtype=data_type)
    text_labels = torch.empty(test_len, device=model_device, dtype=data_type)

    for data_pair_image, i in zip(test_loader, tqdm(range(len(test_loader)))):
        images_id, images, text_id = data_pair_image["images_id"].to(model_device), data_pair_image["images"].to(
            model_device), data_pair_image["text_id"].to(model_device)
        batch_len_image = len(images)
        image_labels[batch_size * i:batch_size * i + batch_len_image] = images_id
        text_labels[batch_size * i:batch_size * i + batch_len_image] = text_id

        for data_pair_text, j in zip(test_loader_iter, range(len(test_loader))):
            text_id, text, attn_mask = data_pair_text["text_id"].to(model_device), data_pair_text["text"].to(
                model_device), data_pair_text["attn_mask"].to(model_device)
            batch_len_text = len(text)

            sim_part = model.validate_forward({
                "images_id": images_id,
                "images": images,
                "text_id": text_id,
                "text": text,
                "attn_mask": attn_mask
            })["sim"]
            sim[batch_size * i:batch_size * i + batch_len_image,
            batch_size * j:batch_size * j + batch_len_text] = sim_part

        torch.cuda.empty_cache()  # free GPU

    return validate_on_tensor(opt, sim, image_labels, text_labels)


def save_log_txt(writer, log: str):
    print(log, end="")
    with open(writer.log_dir + "log.txt", mode="a+", encoding="utf-8") as f:
        f.write(log)


def mark_validate(opt, step: int, metric_pair: dict, best_metric_dict: dict, writer):
    mark = "{:*^100}\n".format(
        time.strftime('[%Y-%m-%d %H:%M:%S] ', time.localtime()) + "validate Epoch: [{:0>3d}/{:0>3d}]".format(
            step, opt["train"]["epoch"])
    )

    img_re = "Now Scores:\n  Text Retrieval: R1:{:<5.2f} R5:{:<5.2f} R10:{:<5.2f} mR:{:<5.2f}".format(
        metric_pair["i2t"]["R1"], metric_pair["i2t"]["R5"], metric_pair["i2t"]["R10"], metric_pair["i2t"]["mR"])
    text_re = "| Image Retrieval: R1:{:<5.2f} R5:{:<5.2f} R10:{:<5.2f} mR:{:<5.2f}".format(
        metric_pair["t2i"]["R1"], metric_pair["t2i"]["R5"], metric_pair["t2i"]["R10"], metric_pair["t2i"]["mR"])
    mR = " | total mR:{:.2f} sum:{:.2f}\n".format(metric_pair["mR"], metric_pair["sum"])
    best_img_re = "Best Scores:\n  Text Retrieval: R1:{:<5.2f} R5:{:<5.2f} R10:{:<5.2f} mR:{:<5.2f}".format(
        best_metric_dict["i2t"]["R1"], best_metric_dict["i2t"]["R5"], best_metric_dict["i2t"]["R10"],
        best_metric_dict["i2t"]["mR"])
    best_text_re = "| Image Retrieval: R1:{:<5.2f} R5:{:<5.2f} R10:{:<5.2f} mR:{:<5.2f}".format(
        best_metric_dict["t2i"]["R1"], best_metric_dict["t2i"]["R5"], best_metric_dict["t2i"]["R10"],
        best_metric_dict["t2i"]["mR"])
    best_mR = " | total mR:{:.2f} sum:{:.2f}\n".format(best_metric_dict["mR"], best_metric_dict["sum"])

    writer.add_scalar("Text Retrieval/R1", scalar_value=metric_pair["i2t"]["R1"], global_step=step)
    writer.add_scalar("Text Retrieval/R5", scalar_value=metric_pair["i2t"]["R5"], global_step=step)
    writer.add_scalar("Text Retrieval/R10", scalar_value=metric_pair["i2t"]["R10"], global_step=step)
    writer.add_scalar("Text Retrieval/mR", scalar_value=metric_pair["i2t"]["R10"], global_step=step)

    writer.add_scalar("Image Retrieval/R1", scalar_value=metric_pair["t2i"]["R1"], global_step=step)
    writer.add_scalar("Image Retrieval/R5", scalar_value=metric_pair["t2i"]["R5"], global_step=step)
    writer.add_scalar("Image Retrieval/R10", scalar_value=metric_pair["t2i"]["R10"], global_step=step)
    writer.add_scalar("Image Retrieval/mR", scalar_value=metric_pair["t2i"]["mR"], global_step=step)

    writer.add_scalar("mR", scalar_value=metric_pair["mR"], global_step=step)

    save_log_txt(writer, mark + img_re + text_re + mR + best_img_re + best_text_re + best_mR)


def generate_word_piece_vocab(json_path, vocab_count_start=5):
    """
    @param json_path: json file path:
    @param vocab_count_start: vocab count start from special character
    @return: vocab
    just adapt for dataset_RSITMD.json
    """
    with open(json_path, "r") as file:
        data = json.load(file)
    count = vocab_count_start
    vocab = {}
    for image_data in data["images"]:
        for sentence in image_data["sentences"]:
            for words in sentence["tokens"]:
                words = words.split("-")
                for word in words:
                    word = word.lower().strip()
                    if word not in vocab:
                        vocab[word] = count
                        count += 1
    return vocab


def generate_tokenizer_json(
        data_json_path,
        base_tokenizer_json_path,
        output_tokenizer_file_path,
):
    """
    @param data_json_path: dataset json file path
    @param base_tokenizer_json_path: base null tokenizer json file path
    @param output_tokenizer_file_path: processed tokenizer json file path
    hugging face bert tokenizer requires json
    """
    with open(base_tokenizer_json_path, "r") as file:
        base_json = json.load(file)
    base_vocab = base_json["model"]["vocab"]
    new_vocab = generate_word_piece_vocab(data_json_path, len(base_vocab))
    base_vocab.update(new_vocab)
    with open(output_tokenizer_file_path + "tokenizer.json", "w") as file:
        json.dump(base_json, fp=file)
    with open(output_tokenizer_file_path + "vocab.txt", "w") as file:
        for key, value in base_vocab.items():
            file.write(key + "\n")
    with open(output_tokenizer_file_path + "tokenizer_config.json", "w") as file:
        config_json = {
            "do_lower_case": True,
            "model_max_length": 512
        }
        json.dump(config_json, fp=file)


# if __name__ == "__main__":
#     generate_tokenizer_json(
#         data_json_path="./dataset/RSITMD/dataset_RSITMD.json",
#         base_tokenizer_json_path="models_data/base_cnn/tokenizer_source/tokenizer.json",
#         output_tokenizer_file_path="models_data/base_cnn/tokenizer/",
#     )
#     tokenizer = BertTokenizer.from_pretrained(r"./models_data/base_cnn/tokenizer/")
#     caption = tokenizer(["this sports field fffff is surrounded by rows of lush trees alongside a wide road"],
#                         return_tensors='pt', padding="max_length",
#                         max_length=30, truncation=True)
#     print(caption)
#     print(tokenizer.decode(caption["input_ids"][0]))
#
#     # with open("./vocab/rsitmd_splits_vocab.json") as f:
#     #     data = json.load(f)["word2idx"]
#     # vocab = generate_word_piece_vocab("./vocab/temp2/dataset_RSITMD.json", 0)
#     # for key, value in vocab.items():
#     #     # print(key)
#     #     if key not in data:
#     #         print(key)
#
#     # print(data)


@torch.no_grad()
def mark_corr_matrix(tensor: torch.Tensor, step: int, writer):
    # value = 0
    # for matrix in tensor:
    #     corr = numpy.abs(numpy.corrcoef(matrix))
    #     value = (numpy.sum(corr) - matrix.shape[0]) / 2 / len(corr)
    values = torch.abs(torch.bmm(tensor, tensor.transpose(1, 2)))
    b, tokens, _ = values.shape
    value = torch.sum(values).item() / (b * tokens * tokens)
    writer.add_scalar("Corr", scalar_value=value / len(tensor), global_step=step)


def generate_random_samples(options, percent=0.8):
    # load all anns
    with open(options['dataset']['path'] + 'caps.txt', 'r', encoding='utf-8') as file:
        caps = file.readlines()
    with open(options['dataset']['path'] + 'filenames.txt', 'r', encoding='utf-8') as file:
        file_names = file.readlines()
    # merge
    assert len(caps) // 5 == len(file_names)

    all_infos = []
    for img_id in range(len(file_names)):
        cap_id = [img_id * 5, (img_id + 1) * 5]
        all_infos.append([caps[cap_id[0]:cap_id[1]], file_names[img_id]])
    random.shuffle(all_infos)

    percent = 0.8
    train_infos = all_infos[:int(len(all_infos) * percent)]
    val_infos = all_infos[int(len(all_infos) * percent):]

    def split_write(data, data_type):
        caps, file_names = [], []
        for item in data:
            caps += item[0]
            file_names.append(item[1])
        with open(options['dataset']['path'] + '{}_caps.txt'.format(data_type), 'w', encoding='utf-8') as file:
            file.writelines(caps)

        with open(options['dataset']['path'] + '{}_filename.txt'.format(data_type), 'w', encoding='utf-8') as file:
            file.writelines(file_names)

    split_write(train_infos, "train")
    split_write(val_infos, "val")

    print("Generate random samples to {} complete.train={} val={}".format(options['dataset']['path'],
                                                                          len(train_infos * 5),
                                                                          len(val_infos * 5)))


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)
