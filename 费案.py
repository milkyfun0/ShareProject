"""
@File  :费案.py
@Author:CaoQiXuan
@Date  :23/12/610:55
@Desc  :
"""
# ----------------------------------------------------utils------------------------------------------
# if __name__ == "__main__":
# opt = get_options()
# train_loader, test_loder = get_loader(opt)
# model = Network(opt=opt).cuda()
# print(validate_with_no_cross(test_loader=test_loder, model=model, opt=opt))
# arr = numpy.load("test.npy")
# def acc_t2i_numpy(sim: numpy.ndarray, true_ids: Optional[numpy.ndarray] = None, img_div: Optional[int] = None):
#     """
#     :param sim: text_len, image_len
#     :param true_ids: text_len, 1
#     :param img_div:
#     :return:
#     des three examples inputs:
#     1. sim None None
#     2. sim None img_div  s.t. text_len // image_len != img_div
#     mark 20231203 这里默认的是同一个image的caption编号相近
#     """
#
#     text_len, image_len = sim.shape
#     scale = img_div if img_div is not None else text_len // image_len
#     if true_ids is None:
#         true_ids = numpy.array(range(text_len)).reshape(-1, 1) // scale
#     else:
#         true_ids.reshape(-1, 1)
#     pre_ids = numpy.argsort(-sim)
#     if (text_len == image_len) and (img_div != 1):
#         ranks = np.where(pre_ids == true_ids)[1] // img_div
#     else:
#         ranks = np.where(pre_ids == true_ids)[1]
#     return metricR(ranks), ranks
#
# def acc_i2t_numpy(
#         sim: numpy.ndarray,
#         img_div: int,
#         true_ids: Optional[numpy.ndarray] = None
# ):
#     """
#     :param sim: image_len, text_len
#     :param true_ids: image_len, 1
#     :param img_div:
#     :return:
#     des three examples inputs:
#     1. sim None None
#     2. sim None img_div  s.t. text_len // image_len != img_div
#     """
#     image_len, text_len = sim.shape
#     scale = img_div if img_div is not None else text_len // image_len
#     if true_ids is None:
#         true_ids = numpy.array(range(image_len)).reshape(-1, 1)
#     else:
#         true_ids.reshape(-1, 1)
#     if (text_len == image_len) and (img_div != 1):
#         true_ids = true_ids // img_div
#     pre_ids = numpy.argsort(-sim) // scale
#     ranks = numpy.where(pre_ids == true_ids)[1].reshape(-1, scale)[:, 0]
#     return metricR(ranks), ranks

# print(numpy.array([(1, 2)]))

# -------------------------------------------- Alignment--------------------------------------------
# image_tokens_id = torch.argsort(image_token_scores, dim=-1, descending=True)[:, :,
#                   :min(self.top_k, patch)]  # b_i, b_t, k
# image_token_mask = torch.scatter(
#     input=torch.zeros((batch_image, batch_text, patch), dtype=torch.int32, device=image_tokens.device),
#     index=image_tokens_id,
#     value=1, dim=-1)  # b_i, b_t, patch
# # test_image_token = t2(image_tokens, image_token_mask).reshape(batch_image, batch_text, -1, dim)
# image_token_mask = image_token_mask.unsqueeze(-1).repeat_interleave(repeats=dim, dim=-1)  # b_i, b_t, patch, dim
# image_tokens = torch.masked_select(input=image_tokens, mask=image_token_mask == 1).reshape(
#     batch_image, batch_text, -1, dim)  # b_i, b_t, k, dim
# text_tokens_id = torch.argsort(text_token_scores, dim=-1, descending=True)[:, :,
#                  :min(self.top_k, s)]  # b_i, b_t, s
# # text_token_mask = torch.scatter(
# #     input=torch.zeros((batch_image, batch_text, s), dtype=torch.int32, device=image_tokens.device),
# #     index=text_tokens_id, value=1, dim=-1) & atte_mask  # b_i, b_t, s
# text_token_mask = torch.scatter(
#     input=torch.zeros((batch_image, batch_text, s), dtype=torch.int32, device=image_tokens.device),
#     index=text_tokens_id, value=1, dim=-1)  # b_i, b_t, s
# text_token_mask = text_token_mask.unsqueeze(-1).repeat_interleave(repeats=dim, dim=-1)  # b_i, b_t, s, dim
# text_tokens = torch.masked_select(input=text_tokens, mask=text_token_mask == 1).reshape(
#     batch_image, batch_text, -1, dim)  # b_i, b_t, k, dim
