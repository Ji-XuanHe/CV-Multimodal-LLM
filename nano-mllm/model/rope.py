"""nano-mllm · D04 旋转位置编码 (RoPE)
对应路线图 D04「位置编码：从 Sinusoidal 到 RoPE」。
运行: python nano-mllm/model/rope.py
依赖: torch
"""
import torch


def apply_rope(x, theta=10000.0):
    """对 (..., T, d) 的最后一维做旋转位置编码；d 必须为偶数。
    用法：对 q、k 各做一次 apply_rope，注意力分数就只依赖相对位置。
    """
    *prefix, T, d = x.shape
    pos = torch.arange(T, device=x.device).float()
    freqs = theta ** (-torch.arange(0, d, 2, device=x.device).float() / d)
    ang = pos[:, None] * freqs[None, :]            # (T, d/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin           # 旋转矩阵作用
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


def _rope_vec(x, pos, theta=10000.0):              # 单向量版，用于相对位置自检
    d = x.size(0)
    i = torch.arange(0, d, 2).float()
    ang = pos * theta ** (-i / d)
    c, s = ang.cos(), ang.sin()
    out = torch.empty_like(x)
    out[0::2] = x[0::2] * c - x[1::2] * s
    out[1::2] = x[0::2] * s + x[1::2] * c
    return out


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k = torch.randn(64), torch.randn(64)
    # 整体平移相同位置（相对距离不变）-> 内积应几乎不变
    da = _rope_vec(q, 2) @ _rope_vec(k, 5)
    db = _rope_vec(q, 4) @ _rope_vec(k, 7)
    print(f"相对距离相同时内积差 = {(da - db).abs().item():.2e}  -> RoPE 只依赖相对位置 ✅")
