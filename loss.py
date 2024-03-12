"""
@File  :loss.py
@Author:CaoQiXuan
@Date  :23/12/820:30
@Desc  :
"""
import torch
from torch import nn


def calc_triple_loss(scores: torch.Tensor, margin: torch.float):
    """
     learn from: Exploring a Fine-Grained Multiscale Method for Cross-Modal Remote Sensing Image Retrieval
    :param scores:
    :param margin:
    :return:
    """
    image_len, text_len = scores.shape
    assert image_len == text_len
    positive = scores.diag().view(-1, 1)
    i2t_positive = positive.expand_as(scores)
    t2i_positive = positive.T.expand_as(scores)
    mask = 1 - torch.eye(image_len, device=scores.device)

    loss_i2t = (margin + scores - i2t_positive).clamp(min=0) * mask
    loss_t2i = (margin + scores - t2i_positive).clamp(min=0) * mask

    # loss = (loss_i2t.sum() + loss_t2i.sum()) / len(loss_t2i)
    loss = loss_i2t.sum() + loss_t2i.sum()

    return loss


def calc_cls_loss(logistic: torch.Tensor, labels: torch.Tensor):
    """
    CrossEntropyLoss; if dataset distribute is not balanced please use focal loss: $-(1-p_t)^\\gamma log(p_t) $
    test ok
    """
    cross_entropy = nn.CrossEntropyLoss(reduction="mean")
    if labels is not torch.LongTensor:
        labels = labels.to(torch.long)
    loss = cross_entropy(logistic, labels.flatten())

    return loss.sum() / len(loss)


def calc_mask_loss(probs, labels):
    loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=-1)
    return loss(probs, labels)


def calc_contrastive_loss(
        feature_row: torch.Tensor,
        feature_col: torch.Tensor,
        label_row: torch.Tensor,
        label_col: torch.Tensor,
        mask_diag: bool = False,
        div: int = 5,
        t: float = 0.1,
        eps: float = 1e-10
):
    """
    contrastive_loss
    :param feature_row: (b, n)
    :param feature_col: (b, n)
    :param label_row: (b, -1)
    :param label_col: (b, -1)
    :param mask_diag:  True: diag is not pair, False: diag is not pair
    :param t: temperature
    :param eps:
    :return:
    """
    assert feature_row.shape == feature_col.shape

    label_col = torch.div(label_col, div, rounding_mode="floor")

    feature_row = feature_row / feature_row.norm(dim=1, keepdim=True)
    feature_col = feature_col / feature_col.norm(dim=1, keepdim=True)
    # print(label_row.reshape(-1, 1) == label_col.reshape(1, -1))
    mask = (label_row.reshape(-1, 1) == label_col.reshape(1, -1)).to(torch.int32)

    mask_diag = (1 - torch.eye(feature_row.shape[0], device=feature_row.device)) if mask_diag else torch.ones_like(
        mask, device=feature_row.device)

    mask = mask * mask_diag
    row_col = feature_row @ feature_col.T / t * mask_diag
    col_row = feature_col @ feature_row.T / t * mask_diag

    row_col_loss = calc_contrastive_loss_part(sim=row_col, mask=mask, eps=eps)
    col_row_loss = calc_contrastive_loss_part(sim=col_row, mask=mask.T, eps=eps)

    return row_col_loss + col_row_loss


def calc_contrastive_loss_part(
        sim: torch.Tensor,
        mask: torch.Tensor,
        eps: float = 1e-10
):
    """
    :param sim: (b, b)
    :param mask: (b, b)
    :param eps:
    :return:
    """

    sim_max, _ = torch.max(sim, dim=1, keepdim=True)
    sim = sim - sim_max
    exp_sim = torch.exp(sim)
    sim = sim * mask

    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + eps)

    mask_sum = mask.sum(dim=1)
    mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

    loss = -1 * (mask * log_prob).sum(dim=1)
    return loss.sum()


def calc_cmpm_loss(
        dual_sim,
        img_div: int,
        image_label: torch.Tensor,
        text_label: torch.Tensor,
        image_feature: torch.Tensor = None,
        text_feature: torch.Tensor = None,
        proj: bool = False,
        t: int = 0.1,
        eps: float = 1e-10
):
    """
    this function need to be tested 20231207
    :param image_feature: (b, n) or
    :param text_feature: (b, n)
    :param image_label: (b, -1)
    :param text_label: (b, -1)
    :param dual_sim:
    :param img_div:
    :param proj:
        True: https://openaccess.thecvf.com/content_ECCV_2018/papers/Ying_Zhang_Deep_Cross-Modal_Projection_ECCV_2018_paper.pdf
        False:  https://arxiv.org/abs/2303.12501
    :param t: temperature
    :param eps:
    :return:
    """
    text_label = torch.div(text_label, img_div, rounding_mode="floor")
    mask = (image_label.reshape(-1, 1) == text_label.reshape(1, -1)).to(torch.float)

    if dual_sim is None:
        image_norm = image_feature / image_feature.norm(dim=-1, keepdim=True)
        text_norm = text_feature / text_feature.norm(dim=-1, keepdim=True)
        image_text = image_feature @ text_norm.T if proj else image_norm @ text_norm.T / t
        text_image = text_feature @ image_norm.T if proj else text_norm @ image_norm.T / t
    else:
        image_text = dual_sim
        text_image = dual_sim.T

    # normalize the true matching distribution, default gaussian distribution
    text_labels_distribute = mask / mask.sum(dim=-1, keepdim=True)
    image_labels_distribute = mask.T / mask.T.sum(dim=-1, keepdim=True)

    i2t_pred = nn.functional.softmax(image_text, dim=-1)
    i2t_loss = i2t_pred * (nn.functional.log_softmax(image_text, dim=-1) - torch.log(text_labels_distribute + eps))

    t2i_pred = nn.functional.softmax(text_image, dim=-1)
    t2i_loss = t2i_pred * (nn.functional.log_softmax(text_image, dim=-1) - torch.log(image_labels_distribute + eps))

    cmpm_loss = torch.sum(torch.sum(i2t_loss, dim=1)) + torch.sum(torch.sum(t2i_loss, dim=1))

    return cmpm_loss / 2
