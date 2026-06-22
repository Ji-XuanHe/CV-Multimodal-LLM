"""nano-mllm · D11 CLIP：对称对比损失 InfoNCE + 单 batch 过拟合自检
对应路线图 D11「CLIP」。对比学习第一件事永远是「单 batch 过拟合自检」：
连一个 batch 都学不到对角线 100%，代码一定有 bug。
运行: python nano-mllm/vision/clip.py   （纯 torch，无需下载预训练权重）
依赖: torch
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_contrastive_loss(img_emb, txt_emb, logit_scale):
    img = F.normalize(img_emb, dim=-1)            # L2 归一化（漏了它 loss 不降！）
    txt = F.normalize(txt_emb, dim=-1)
    logits = logit_scale.exp() * img @ txt.t()    # (N, N) 相似度矩阵
    labels = torch.arange(img.size(0))            # 对角线才是匹配对
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    return (loss_i2t + loss_t2i) / 2, logits


if __name__ == "__main__":
    torch.manual_seed(0)
    # 用两个小 encoder 模拟图像塔 / 文本塔（这里不下载 timm，纯演示对比损失机制）
    img_enc = nn.Linear(64, 128)
    txt_enc = nn.Linear(32, 128)
    logit_scale = nn.Parameter(torch.tensor(2.6593))   # ln(1/0.07)，CLIP 初值
    opt = torch.optim.AdamW(list(img_enc.parameters()) + list(txt_enc.parameters()) + [logit_scale], lr=1e-2)

    imgs = torch.randn(8, 64)
    txts = torch.randn(8, 32)
    for step in range(301):
        loss, logits = clip_contrastive_loss(img_enc(imgs), txt_enc(txts), logit_scale)
        opt.zero_grad()
        loss.backward()
        opt.step()
        logit_scale.data.clamp_(max=math.log(100))     # 防温度爆炸（CLIP 同款，更新后再 clamp）
        if step % 100 == 0:
            acc = (logits.argmax(-1) == torch.arange(8)).float().mean().item()
            print(f"step {step:3d}  loss {loss.item():.3f}  in-batch acc {acc:.2f}")
    print("单 batch 过拟合到对角线 100% -> 对比损失实现正确 ✅")
