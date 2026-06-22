"""nano-mllm · P03/D01 注意力：Scaled Dot-Product + Multi-Head + Causal(+RoPE)
对应路线图 P03「注意力机制」、D01「Causal Attention」。
整个项目复用最多的文件：ViT / GPT / CLIP 文本塔 / LLaVA 都建立在它之上。
运行: python nano-mllm/model/attention.py
依赖: torch
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo 根
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import apply_rope


class CausalSelfAttention(nn.Module):
    """因果多头自注意力（GPT 的核心积木），可选 RoPE。"""

    def __init__(self, d_model, n_heads, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.use_rope = use_rope

    def forward(self, x):                                  # x: (B, T, d)
        B, T, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q = q.view(B, T, self.h, self.dk).transpose(1, 2)  # (B, h, T, dk)
        k = k.view(B, T, self.h, self.dk).transpose(1, 2)
        v = v.view(B, T, self.h, self.dk).transpose(1, 2)
        if self.use_rope:
            q, k = apply_rope(q), apply_rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # 内置走 FlashAttention
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.proj(out)


def manual_attention(Q, K, V):
    """手写 scaled dot-product（无 mask），用于和官方 SDPA 对拍；causal 路径由 CausalSelfAttention 的 is_causal=True 负责。"""
    S = Q @ K.transpose(-2, -1) / Q.size(-1) ** 0.5
    return F.softmax(S, dim=-1) @ V


if __name__ == "__main__":
    torch.manual_seed(0)
    Q, K, V = (torch.randn(2, 8, 16, 64) for _ in range(3))
    diff = (manual_attention(Q, K, V) - F.scaled_dot_product_attention(Q, K, V)).abs().max()
    print(f"手写 vs 官方 SDPA max diff = {diff.item():.2e}  -> 实现正确 ✅")

    attn = CausalSelfAttention(d_model=128, n_heads=4)
    print("CausalSelfAttention 输出:", tuple(attn(torch.randn(1, 10, 128)).shape))
