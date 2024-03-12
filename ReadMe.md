

# Base_Clip

## 20240222 PatchSzie大小实验

采用原始架构TinyCLip进行实验

实验结果：

**参数**：**lr=0.0002**, **magin=0.15**

```python
# TinyCLIP-ViT-61M-32-Text-29M-LAION400M
# Text Retrieval: R1:15.00 R5:33.12 R10:44.13 mR:30.75| Image Retrieval: R1:12.58 R5:36.10 R10:51.70 mR:33.46 | total mR:32.10 sum:192.63  not freeze
# Text Retrieval: R1:13.50 R5:30.75 R10:41.38 mR:28.54| Image Retrieval: R1:11.23 R5:34.00 R10:49.53 mR:31.58 | total mR:30.06 sum:180.38   freeze 6 and 4
# Text Retrieval: R1:13.50 R5:29.75 R10:41.50 mR:28.25| Image Retrieval: R1:9.78  R5:32.35 R10:48.08 mR:30.07 | total mR:29.16 sum:174.95   freeze 9 and 6


# TinyCLIP-ViT-39M-16-Text-19M-YFCC15M 12 and 6
# Text Retrieval: R1:16.75 R5:34.50 R10:45.63 mR:32.29| Image Retrieval: R1:12.65 R5:37.18 R10:53.48 mR:34.43 | total mR:33.36 sum:200.18
# Text Retrieval: R1:15.50 R5:32.62 R10:43.13 mR:30.42| Image Retrieval: R1:12.45 R5:35.30 R10:50.38 mR:32.71 | total mR:31.56 sum:189.38 freeze 6 and 3
# Text Retrieval: R1:15.50 R5:30.75 R10:43.75 mR:30.00| Image Retrieval: R1:12.00 R5:34.68 R10:49.50 mR:32.06 | total mR:31.03 sum:186.18 freeze 8 and 4

# TinyCLIP-ViT-40M-32-Text-19M-LAION400M 12 and 6
# Text Retrieval R1:15.75 R5:31.63 R10:42.75 mR:30.04| Image Retrieval: R1:11.50 R5:35.08 R10:50.45 mR:32.34 | total mR:31.19 sum:187.15
```

```python
# [2024-03-02 12:49:24] Epoch: [000/020] [0/268] total loss: 728.33 time: 6.49 s
# [2024-03-02 12:50:00] Epoch: [000/020] [53/268] total loss: 109.72 time: 35.76s
# [2024-03-02 12:50:36] Epoch: [000/020] [106/268] total loss: 68.37 time: 36.11s
# [2024-03-02 12:51:12] Epoch: [000/020] [159/268] total loss: 75.89 time: 36.19s
# [2024-03-02 12:51:48] Epoch: [000/020] [212/268] total loss: 91.77 time: 36.15s
# [2024-03-02 12:52:24] Epoch: [000/020] [265/268] total loss: 61.24 time: 36.15s
# **************************[2024-03-02 12:52:46] validate Epoch: [000/020]***************************
# Now Scores:
#   Text Retrieval: R1:11.75 R5:27.25 R10:39.88 mR:26.29| Image Retrieval: R1:11.35 R5:33.12 R10:49.20 mR:31.23 | total mR:28.76 sum:172.55
# Best Scores:
#   Text Retrieval: R1:11.75 R5:27.25 R10:39.88 mR:26.29| Image Retrieval: R1:11.35 R5:33.12 R10:49.20 mR:31.23 | total mR:28.76 sum:172.55
```



## 20240229 添加了向量相似化分析

![image-20240308113559663](D:\Code\Pycharm\Project\figures\20240229.png)

## 20240303 添加了AttentionMap可视化

![image-20240308113713568](D:\Code\Pycharm\Project\figures\20240303png)



# Base_CNN

## 20240224 编写该模块

**Visual**：Resnet34

**Text**：Bert

实验结果：无法收敛

# Base_Register

## 202402026 加入Register模块

```python
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
```

好像没啥卵用



```python
# [2024-03-08 22:12:23] Epoch: [000/020] [0/269] total loss: 1209.44 time: 7.28 s
# [2024-03-08 22:12:48] Epoch: [000/020] [53/269] total loss: 714.87 time: 25.18s
# [2024-03-08 22:13:13] Epoch: [000/020] [106/269] total loss: 378.44 time: 25.20s
# [2024-03-08 22:13:39] Epoch: [000/020] [159/269] total loss: 419.25 time: 25.25s
# [2024-03-08 22:14:04] Epoch: [000/020] [212/269] total loss: 267.92 time: 25.20s
# [2024-03-08 22:14:29] Epoch: [000/020] [265/269] total loss: 158.88 time: 25.23s
#  94%|█████████▍| 16/17 [00:16<00:01,  1.03s/it]
# **************************[2024-03-08 22:14:47] validate Epoch: [000/020]***************************
# Now Scores:
#   Text Retrieval: R1:3.73  R5:10.71 R10:18.39 mR:10.94| Image Retrieval: R1:3.96  R5:15.06 R10:24.98 mR:14.67 | total mR:12.81 sum:76.83
# Best Scores:
#   Text Retrieval: R1:3.73  R5:10.71 R10:18.39 mR:10.94| Image Retrieval: R1:3.96  R5:15.06 R10:24.98 mR:14.67 | total mR:12.81 sum:76.83


# Now Scores:
#  Text Retrieval: R1:6.64  R5:19.67 R10:29.22 mR:18.51| Image Retrieval: R1:7.19  R5:22.37 R10:35.27 mR:21.61 | total mR:20.06 sum:120.37
#Best Scores:
#  Text Retrieval: R1:7.22  R5:19.32 R10:30.38 mR:18.98| Image Retrieval: R1:6.64  R5:21.91 R10:35.53 mR:21.36 | total mR:20.17 sum:121.00
```



# Base_Trans

## 20240226 更改了文本编码器

**Text：**将vocab下降

不太行

# Base_PosCNN

![image-20240308222144765](D:\Code\Pycharm\Project\figures\PosCNN)

```python
class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim, stride=2):
        super(PosCNN, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_chans, embed_dim, 3, stride, 1, bias=True, groups=embed_dim // 8))
        self.stride = stride
        self.ln_1 = LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            QuickGELU(),
            # nn.Linear(embed_dim * 4, embed_dim),
            LayerNorm(embed_dim)
        )

    def tokens_attention(self, tokens):
        att = torch.mean(tokens, dim=2, keepdim=True)
        tokens = tokens * torch.sigmoid(att)
        return self.mlp(tokens)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        cls_token = x[:, 0, :].unsqueeze(1)
        x = x[:, 1:, :]

        B, N, C = x.shape
        H, W = int(math.sqrt(N)), int(math.sqrt(N))
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)

        tokens = self.proj(cnn_feat).flatten(2).transpose(1, 2)
        tokens = tokens + self.tokens_attention(tokens)

        x = torch.cat((cls_token, tokens), dim=1)

        x = x.permute(1, 0, 2)
        x = self.ln_1(x)
        return x
```

效果一般

